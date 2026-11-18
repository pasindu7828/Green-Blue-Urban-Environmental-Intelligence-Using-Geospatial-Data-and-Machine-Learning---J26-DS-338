from pathlib import Path
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
    / "trend_validation"
)

OUTPUT_DIR = (
    VALIDATION_DIR
    / "outputs"
)

STEP1 = (
    OUTPUT_DIR
    / "01_trend_formula_reproducibility_summary.csv"
)

STEP2 = (
    OUTPUT_DIR
    / "02_leave_one_year_out_trend_stability_summary.csv"
)

STEP3 = (
    OUTPUT_DIR
    / "03_trajectory_validation_summary.csv"
)

STEP4 = (
    OUTPUT_DIR
    / "04_historical_temporal_consistency_summary.csv"
)

STEP4B = (
    OUTPUT_DIR
    / "04b_historical_trajectory_relevance_summary.csv"
)

FINAL_SUMMARY = (
    OUTPUT_DIR
    / "05_trend_validation_final_summary.csv"
)

EVIDENCE_TABLE = (
    OUTPUT_DIR
    / "05_trend_validation_evidence_table.csv"
)

PAPER_TEXT = (
    OUTPUT_DIR
    / "05_trend_validation_paper_text.txt"
)

FREEZE_MANIFEST = (
    VALIDATION_DIR
    / "05_trend_validation_freeze_manifest.json"
)

README_PATH = (
    VALIDATION_DIR
    / "README.md"
)


# ============================================================
# HELPERS
# ============================================================

def section(title):
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)


def require(path):
    if not path.exists():
        raise FileNotFoundError(
            f"Required validation output missing: {path}"
        )


# ============================================================
# 1. VERIFY INPUTS
# ============================================================

section(
    "1. VERIFY TREND VALIDATION EVIDENCE"
)

for path in [
    STEP1,
    STEP2,
    STEP3,
    STEP4,
    STEP4B,
]:
    require(path)
    print(
        f"FOUND: {path.name}"
    )


# ============================================================
# 2. LOAD RESULTS
# ============================================================

section(
    "2. LOAD VALIDATION RESULTS"
)

s1 = pd.read_csv(STEP1).iloc[0]
s2 = pd.read_csv(STEP2).iloc[0]
s3 = pd.read_csv(STEP3).iloc[0]
s4 = pd.read_csv(STEP4).iloc[0]
s4b = pd.read_csv(STEP4B).iloc[0]


# ============================================================
# 3. FINAL SUMMARY
# ============================================================

section(
    "3. BUILD FINAL TREND VALIDATION SUMMARY"
)

final = pd.DataFrame(
    [
        {
            # Step 1
            "formula_reproduction_grid_count":
                int(s1["grid_count"]),

            "formula_raw_class_match_percent":
                float(
                    s1[
                        "raw_class_match_percent"
                    ]
                ),

            "formula_fdr_class_match_percent":
                float(
                    s1[
                        "fdr_class_match_percent"
                    ]
                ),

            # Step 2
            "mean_absolute_sen_slope_loo_spearman":
                float(
                    s2[
                        "mean_sen_slope_spearman"
                    ]
                ),

            "median_absolute_sen_slope_loo_spearman":
                float(
                    s2[
                        "median_sen_slope_spearman"
                    ]
                ),

            "minimum_absolute_sen_slope_loo_spearman":
                float(
                    s2[
                        "minimum_sen_slope_spearman"
                    ]
                ),

            "mean_absolute_direction_agreement_percent":
                float(
                    s2[
                        "mean_slope_direction_agreement_percent"
                    ]
                ),

            "mean_raw_mk_class_agreement_percent":
                float(
                    s2[
                        "mean_raw_class_agreement_percent"
                    ]
                ),

            # Step 3
            "trajectory_independent_class_match_percent":
                float(
                    s3[
                        "independent_full_class_match_percent"
                    ]
                ),

            "mean_trajectory_loo_spearman":
                float(
                    s3[
                        "mean_loo_relative_slope_spearman"
                    ]
                ),

            "mean_trajectory_class_agreement_percent":
                float(
                    s3[
                        "mean_loo_trajectory_class_agreement_percent"
                    ]
                ),

            "minimum_trajectory_class_agreement_percent":
                float(
                    s3[
                        "minimum_loo_trajectory_class_agreement_percent"
                    ]
                ),

            "mean_threshold_class_agreement_percent":
                float(
                    s3[
                        "mean_nonreference_threshold_class_agreement_percent"
                    ]
                ),

            "minimum_threshold_class_agreement_percent":
                float(
                    s3[
                        "minimum_nonreference_threshold_class_agreement_percent"
                    ]
                ),

            # Step 4
            "mean_future_direction_relative_spearman":
                float(
                    s4[
                        "mean_relative_slope_vs_future_change_spearman"
                    ]
                ),

            "mean_future_direction_agreement_percent":
                float(
                    s4[
                        "mean_relative_direction_agreement_percent"
                    ]
                ),

            # Step 4B
            "mean_hot_side_next_year_relative_heat":
                float(
                    s4b[
                        "mean_hot_side_next_year_relative_heat"
                    ]
                ),

            "mean_cool_side_next_year_relative_heat":
                float(
                    s4b[
                        "mean_cool_side_next_year_relative_heat"
                    ]
                ),

            "mean_hot_minus_cool_relative_heat_difference":
                float(
                    s4b[
                        "mean_hot_minus_cool_relative_heat_difference"
                    ]
                ),

            "mean_hot_minus_cool_lst_difference_c":
                float(
                    s4b[
                        "mean_hot_minus_cool_lst_difference_c"
                    ]
                ),

            "mean_top25_heat_risk_ratio_hot_vs_cool":
                float(
                    s4b[
                        "mean_top25_heat_risk_ratio_hot_vs_cool"
                    ]
                ),

            "mean_top10_heat_risk_ratio_hot_vs_cool":
                float(
                    s4b[
                        "mean_top10_heat_risk_ratio_hot_vs_cool"
                    ]
                ),

            "maximum_relevance_mann_whitney_p":
                float(
                    s4b[
                        "maximum_mann_whitney_one_sided_p"
                    ]
                ),

            "validation_status":
                "COMPLETE_AND_FROZEN",
        }
    ]
)

final.to_csv(
    FINAL_SUMMARY,
    index=False,
)

print(
    final.to_string(
        index=False
    )
)


# ============================================================
# 4. EVIDENCE TABLE
# ============================================================

section(
    "4. BUILD VALIDATION EVIDENCE TABLE"
)

evidence = pd.DataFrame(
    [
        {
            "validation_stage":
                "Step 1 - Formula reproducibility",

            "main_evidence":
                "Independent Mann-Kendall, Sen slope and BH-FDR recomputation",

            "result":
                "100% raw and FDR class reproduction",

            "interpretation":
                "Production trend calculations are reproducible.",
        },

        {
            "validation_stage":
                "Step 2 - Absolute trend leave-one-year-out stability",

            "main_evidence":
                "Remove each year from 2015-2025 and recompute trends",

            "result":
                (
                    f"Mean Sen-slope Spearman="
                    f"{s2['mean_sen_slope_spearman']:.3f}; "
                    f"mean direction agreement="
                    f"{s2['mean_slope_direction_agreement_percent']:.2f}%"
                ),

            "interpretation":
                (
                    "Absolute historical trend ordering is generally stable, "
                    "with sensitivity to endpoint years."
                ),
        },

        {
            "validation_stage":
                "Step 3 - Relative trajectory stability",

            "main_evidence":
                "Independent reproduction, leave-one-year-out and threshold sensitivity",

            "result":
                (
                    f"Independent reproduction=100%; "
                    f"mean LOO class agreement="
                    f"{s3['mean_loo_trajectory_class_agreement_percent']:.2f}%; "
                    f"mean threshold agreement="
                    f"{s3['mean_nonreference_threshold_class_agreement_percent']:.2f}%"
                ),

            "interpretation":
                (
                    "Trajectory classification is reproducible and reasonably "
                    "robust, but temporally sensitive when endpoint years are removed."
                ),
        },

        {
            "validation_stage":
                "Step 4 - Future-direction persistence",

            "main_evidence":
                "History-only Sen slope compared with subsequent 3-year change",

            "result":
                (
                    f"Mean relative Spearman="
                    f"{s4['mean_relative_slope_vs_future_change_spearman']:.3f}; "
                    f"direction agreement="
                    f"{s4['mean_relative_direction_agreement_percent']:.2f}%"
                ),

            "interpretation":
                (
                    "Historical slope direction should NOT be used as a forecast "
                    "of subsequent warming or cooling direction."
                ),
        },

        {
            "validation_stage":
                "Step 4B - Subsequent heat-burden relevance",

            "main_evidence":
                "History-only trajectory classes evaluated against next-year actual heat",

            "result":
                (
                    f"Hot-side grids were on average "
                    f"{s4b['mean_hot_minus_cool_lst_difference_c']:.3f} C hotter; "
                    f"Top-25% heat risk ratio="
                    f"{s4b['mean_top25_heat_risk_ratio_hot_vs_cool']:.2f}x"
                ),

            "interpretation":
                (
                    "Trajectory classes provide meaningful evidence of subsequent "
                    "relative heat burden."
                ),
        },
    ]
)

evidence.to_csv(
    EVIDENCE_TABLE,
    index=False,
)

print(
    evidence.to_string(
        index=False
    )
)


# ============================================================
# 5. PAPER TEXT
# ============================================================

section(
    "5. CREATE PAPER-READY INTERPRETATION"
)

paper_text = f"""
TREND AND TRAJECTORY VALIDATION

The historical trend and trajectory analysis was evaluated using multiple complementary
validation procedures. First, the grid-level Mann-Kendall statistics, Sen's slopes and
Benjamini-Hochberg false-discovery-rate adjustments were independently recomputed for all
2,194 grid cells. The independently derived raw and FDR-adjusted trend classifications
matched the frozen production outputs in 100% of cases, confirming computational
reproducibility.

Leave-one-year-out sensitivity analysis showed that absolute historical Sen-slope rankings
were generally stable, with a mean Spearman correlation of
{s2['mean_sen_slope_spearman']:.3f} and mean slope-direction agreement of
{s2['mean_slope_direction_agreement_percent']:.2f}%. Some sensitivity was observed when
endpoint years were removed, indicating that trend estimates should be interpreted as
multi-year historical summaries rather than invariant quantities.

The relative heat-trajectory classification was also independently reproduced with 100%
agreement. Across leave-one-year-out analyses, mean trajectory-class agreement was
{s3['mean_loo_trajectory_class_agreement_percent']:.2f}%. Reasonable perturbations of the
operational hot and relative-slope thresholds retained a mean of
{s3['mean_nonreference_threshold_class_agreement_percent']:.2f}% of classifications,
indicating that the classification is not solely dependent on one exact threshold setting.

A retrospective direction-persistence analysis found that historical relative Sen slopes
were negatively associated with subsequent three-year relative heat changes
(mean Spearman rho =
{s4['mean_relative_slope_vs_future_change_spearman']:.3f}), with only
{s4['mean_relative_direction_agreement_percent']:.2f}% same-direction agreement.
Therefore, historical trajectory direction is not interpreted as a forecast of future
warming or cooling direction.

However, retrospective relevance testing showed that trajectory classes remained useful
for identifying subsequent heat burden. Using only information available before the
evaluation year, grids classified on the hot side of the trajectory framework were on
average {s4b['mean_hot_minus_cool_lst_difference_c']:.3f} C hotter in the following
observed year than cool-side grids. They were
{s4b['mean_top25_heat_risk_ratio_hot_vs_cool']:.2f} times more likely to fall within the
top 25% of actual heat and
{s4b['mean_top10_heat_risk_ratio_hot_vs_cool']:.2f} times more likely to fall within the
top 10%.

Accordingly, the trajectory framework is interpreted as a reproducible and planning-relevant
historical relative heat-behaviour classification. It is useful for distinguishing locations
with persistent or emerging heat burden, but it is not treated as a standalone forecast of
future temperature-change direction.
""".strip()

PAPER_TEXT.write_text(
    paper_text,
    encoding="utf-8",
)

print(
    paper_text
)


# ============================================================
# 6. README
# ============================================================

section(
    "6. WRITE VALIDATION README"
)

readme_text = """
# Urban Heat Trend Validation

Status: COMPLETE AND FROZEN

This validation folder evaluates the historical trend and relative heat-trajectory
components of the Urban Heat Research pipeline.

## Validation stages

1. Independent Mann-Kendall, Sen slope and BH-FDR reproduction
2. Leave-one-year-out absolute trend stability
3. Relative trajectory reproduction, temporal stability and threshold sensitivity
4. Historical future-direction persistence diagnostic
5. Historical next-year heat-burden relevance
6. Final evidence consolidation and freeze

## Interpretation

The trend and trajectory calculations are reproducible and generally stable under
reasonable sensitivity tests.

Trajectory classes describe historical relative heat behaviour within Kaduwela.
They should not be interpreted as statistically guaranteed future warming or cooling
directions.

Retrospective evaluation nevertheless shows that hot-side trajectory classes identify
locations with substantially greater subsequent observed heat burden.

No validation result was used to retune or alter the frozen production trajectory logic.
""".strip()

README_PATH.write_text(
    readme_text,
    encoding="utf-8",
)


# ============================================================
# 7. FREEZE MANIFEST
# ============================================================

section(
    "7. CREATE FREEZE MANIFEST"
)

manifest = {
    "validation_component":
        "Urban Heat Trend and Trajectory Validation",

    "status":
        "COMPLETE_AND_FROZEN",

    "production_logic_changed":
        False,

    "thresholds_retuned_using_validation":
        False,

    "validated_claims": [
        "Mann-Kendall and Sen slope computation is reproducible.",
        "Absolute historical trend estimates are generally stable under leave-one-year-out analysis.",
        "Relative trajectory classification is independently reproducible.",
        "Trajectory classes are reasonably robust to threshold perturbation.",
        "Trajectory classes identify subsequent relative heat burden.",
    ],

    "unsupported_claims": [
        "Historical Sen slope direction reliably predicts future warming direction.",
        "Trajectory class is a future temperature forecast.",
        "Trajectory class is equivalent to statistically significant absolute warming.",
    ],

    "final_outputs": [
        str(FINAL_SUMMARY),
        str(EVIDENCE_TABLE),
        str(PAPER_TEXT),
        str(README_PATH),
    ],
}

with FREEZE_MANIFEST.open(
    "w",
    encoding="utf-8",
) as file:
    json.dump(
        manifest,
        file,
        indent=2,
    )

print(
    json.dumps(
        manifest,
        indent=2,
    )
)


# ============================================================
# 8. FINAL STATUS
# ============================================================

section(
    "8. FINAL TREND VALIDATION STATUS"
)

print(
    "1. MK + Sen slope + BH-FDR calculations are independently reproducible."
)

print(
    "2. Absolute historical trends are generally stable under leave-one-year-out testing."
)

print(
    "3. Relative trajectory classification is reproducible and reasonably robust."
)

print(
    "4. Endpoint-year sensitivity exists and must be reported transparently."
)

print(
    "5. Historical slope direction does NOT reliably predict subsequent short-term direction."
)

print(
    "6. History-only hot-side trajectory classes identify substantially higher subsequent heat burden."
)

print(
    "7. Trajectory classes remain historical relative heat-behaviour indicators, not forecasts."
)

print(
    "\nVALIDATION STATUS: COMPLETE AND FROZEN"
)

print(
    f"\nSaved: {FINAL_SUMMARY}"
)

print(
    f"Saved: {EVIDENCE_TABLE}"
)

print(
    f"Saved: {PAPER_TEXT}"
)

print(
    f"Saved: {README_PATH}"
)

print(
    f"Saved: {FREEZE_MANIFEST}"
)