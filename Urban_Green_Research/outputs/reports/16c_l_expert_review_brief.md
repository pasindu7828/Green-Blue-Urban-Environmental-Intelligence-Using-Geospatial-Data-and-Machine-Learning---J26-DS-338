# Expert Review Request

## Research component

Urban Green Transition, Urbanization Pressure
and Conservation Intelligence — J26-DS-338.

Study area: Kaduwela Municipal Council.

Analysis: 9,240 grid cells, nominal 100 m resolution.

## Purpose

Review the candidate criteria for a relative
general urban development suitability map using
GIS-MCDA and AHP.

This does not determine legal building permission.

## Candidate criteria

| Code | Criterion | Working direction |
|---|---|---|
| C1 | Distance to major roads | Decreasing |
| C2 | Road density, 500 m | Increasing, provisional |
| C3 | Building context, 500 m | Undecided |
| C4 | Slope in degrees | Decreasing |
| C5 | Elevation in metres | Undecided |

## Important research findings

The original raw criteria were validated across
9,240 unique grid cells.

C3 direction is sensitive:

- Percentile-capped: Spearman 0.4775;
  top-decile Jaccard 0.0763.
- Percentile-rank: Spearman 0.5544;
  top-decile Jaccard 0.1748.

These statistical results cannot determine which
C3 direction is scientifically correct.

The 500 m C2 and C3 neighbourhood scales remain
provisional spatial-scale choices.

C5 elevation must not receive a city-wide direction
solely from elevation rules applying to specific
conservation zones.

The provisional UDA GIS is image-derived, has
low-confidence classifications and is legally
unverified. It must not be treated as official
development permission.

Road distance is not direct parcel access.
Road density is not proof of road capacity.
Building density does not show vacant land.
Slope does not replace site engineering studies.

## Requested expert review

First complete:
16c_l_expert_criteria_review_template.csv

For each criterion, provide:

- INCLUDE, EXCLUDE, REVISE or UNCERTAIN.
- INCREASING, DECREASING, NON_MONOTONIC,
  NOT_APPLICABLE or UNCERTAIN for direction.
- A short justification.
- Actual reviewer name, role and review date.

Next, complete the appropriate alternative in:
16c_l_ahp_pairwise_template.csv

Available alternatives:

BASE_3_CRITERIA: C1, C2 and C4.

CONDITIONAL_4_WITH_C3: C1, C2, C3 and C4.

Complete only the alternative matching the
reviewed criterion decision.

If a different criterion set is approved,
the questionnaire must be regenerated.

For each pairwise comparison:

Higher importance: A, B or EQUAL.

Strength: integer 1 to 9.

AHP verbal scale:

1 = Equal importance.
3 = Moderate importance.
5 = Strong importance.
7 = Very strong importance.
9 = Extreme importance.

2, 4, 6 and 8 = Intermediate judgments.

If equal, enter EQUAL and 1.

No reciprocal fractions are required.
The later AHP implementation will calculate them.

## Methodological safeguards

Expert input must be genuinely collected and
recorded. Do not invent responses.

AHP consistency will be validated separately.
Inconsistent judgments may require review.

If genuine expert input cannot be obtained,
a transparently documented alternative using
published evidence or scenario analysis is needed.

Planning screening and legal zoning verification
remain separate from the suitability score.

No final AHP weights, criterion set or suitability
recommendations are approved by this document.
