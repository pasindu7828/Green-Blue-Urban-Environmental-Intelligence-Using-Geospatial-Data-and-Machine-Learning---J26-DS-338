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