# Priority Validation

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
500 times.

- Minimum weight-rank Spearman:
  1.0000
- Top-weight criterion retained:
  100.0%

Interpretation:
CRITIC weight ordering was highly stable under grid resampling.

### Step 2 - TOPSIS Rank Stability Under Resampled CRITIC Weights

TOPSIS was recomputed
500 times.

- Minimum Spearman:
  0.9966
- Median Spearman:
  0.9999
- Minimum Top-100 overlap:
  94.0%
- Median Top-100 overlap:
  98.0%

Interpretation:
The priority ranking remained highly stable to sampling-driven CRITIC
weight variation.

### Existing Step 33 - Weight Sensitivity

Existing production sensitivity analysis includes:

- 24 controlled OAT scenarios
- 500 simultaneous random-weight scenarios
- equal-weight benchmark

Controlled OAT minimum Spearman:
0.9769

Monte Carlo minimum Spearman:
0.9427

Monte Carlo median Spearman:
0.9940

### Step 3 - Criterion Removal

C6 Modeled Intervention Benefit was tested conservatively because scenario
validation requires cautious interpretation.

Removing C6:

- Spearman:
  0.9987
- Top-100 overlap:
  95.0%
- Mean absolute rank shift:
  24.92

Recomputing CRITIC without C6:

- Spearman:
  0.9990
- Top-100 overlap:
  97.0%

Therefore, the final priority ranking is not materially dependent on the
scenario-derived intervention-benefit criterion.

The strongest criterion-removal effect occurred for:

- Criterion:
  C5
- Dimension:
  Blue-space / Water Deficit
- Spearman after removal:
  0.5449
- Top-100 overlap:
  31.0%

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
