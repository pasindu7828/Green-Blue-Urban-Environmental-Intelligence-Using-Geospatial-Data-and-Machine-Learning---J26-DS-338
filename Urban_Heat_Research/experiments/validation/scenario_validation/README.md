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

- +5 percentage points: 73.36% historical support
- +10 percentage points: 55.24% historical support
- +20 percentage points: 32.19% historical support

Built-up scenario levels were independently reproduced using historical
absolute annual NDBI-change quantiles.

Blue-space scenario levels were independently reproduced using historical
positive annual NDWI-change quantiles.

Status: PASS

## Step 2 - Historical Response Consistency

Two-way fixed-effects models were used to examine within-grid historical
associations while controlling for grid and year effects.

Full-period coefficients:

- Green cover beta: 1.971988
- NDBI beta: 13.279485
- NDWI beta: -14.565101

The response signs remained stable when 2017 was excluded.

Interpretation:

- Higher NDBI was consistently associated with higher LST.
- Higher NDWI was consistently associated with lower LST.
- Green-cover change showed a positive observational association with LST
  in this dataset and therefore green interventions must not be described
  as guaranteed cooling.

Status: PASS

## Step 3 - Scenario Output Robustness

Structural, arithmetic, intervention-bound, prediction-difference,
cooling-benefit, direction-label and combined-scenario checks were performed.

- Passed audit checks: 45/45
- Historical-direction agreement: PASS

Monotonicity was evaluated only as a diagnostic and was not forced into the
nonlinear LightGBM scenario model.

## Overall Conclusion

Overall status: PASS_WITH_INTERPRETATION_CAUTION

The scenario framework is supported by historical intervention magnitudes,
stable historical response directions, and internally consistent scenario
outputs.

However, the scenario outputs remain model-based what-if sensitivity
estimates. They should not be interpreted as causal effects, guaranteed
cooling outcomes, or validated future observations.
