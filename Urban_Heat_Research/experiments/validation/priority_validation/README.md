# Cooling-Priority Validation

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
- Minimum Spearman = 0.9966
- Median Spearman = 0.9999
- Minimum Top-100 overlap = 94.0%

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

- Criterion: C5
- Dimension: Blue-space / Water Deficit
- Spearman after removal:
  0.5449
- Top-100 overlap:
  31.0%

This is reported as criterion influence rather than model failure.

### C6 Intervention-Benefit Sensitivity

Removing C6 entirely:

- Spearman = 0.9987
- Top-100 overlap = 95.0%

Therefore, the final ranking is not materially dependent on the
scenario-derived intervention-benefit criterion.

## Validation 05 - Historical Retrospective Backtesting

Two leakage-controlled historical backtests were conducted.

### 2023 -> 2024

The ranking was constructed using information available through 2023 only.

- Priority vs actual 2024 heat Spearman:
  0.4411
- Top-10% priority vs actual hottest Top-10% overlap:
  25.45%
- Top-25% overlap:
  48.09%
- Highest-priority decile minus lowest-priority decile actual LST:
  +3.32 C

### 2024 -> 2025

The ranking was constructed using information available through 2024 only.

- Priority vs actual 2025 heat Spearman:
  0.3732
- Top-10% priority vs actual hottest Top-10% overlap:
  20.91%
- Top-25% overlap:
  46.27%
- Highest-priority decile minus lowest-priority decile actual LST:
  +2.82 C

### Combined Historical Evidence

- Mean priority vs later observed heat Spearman:
  0.4072
- Mean Top-10% overlap:
  23.18%
- Mean Top-25% overlap:
  47.18%
- Mean high-priority vs low-priority LST separation:
  +3.07 C

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
