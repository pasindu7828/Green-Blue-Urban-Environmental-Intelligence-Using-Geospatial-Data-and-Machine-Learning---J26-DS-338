
"""
J26-DS-338 | Section 04 | Stage 4.15
Conservation Priority Sensitivity and Robustness Analysis.

APPROVED AND FROZEN:
    CP = Frozen Green Importance * Frozen Future Pressure

Test alternatives without changing that primary methodology.

All rank comparisons use the SAME 4,609-cell common domain.
Alternative CRITIC weighting uses all 8,727 GI-eligible cells.

Tests:
- Equal G1/G2/G3 weights
- KMC clipped-area-weighted CRITIC
- Six +/-10% relative CRITIC weight perturbations
- Alternative current green lower-bound criterion
- Alternative historical mean green criterion
- Alternative 500m neighbourhood criterion
- Illustrative equal-tradeoff additive combination

Outputs:
  17n_priority_sensitivity_scenarios.csv
  17n_priority_cell_rank_instability.csv
  17n_priority_sensitivity_qa.json

This script does NOT modify frozen GI, frozen pressure, or primary CP.
"""

from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, kendalltau


ROOT = Path(__file__).resolve().parents[2]

CI = ROOT / "data/processed/conservation_intelligence"
REPORT = ROOT / "outputs/reports/conservation_intelligence"

PRIMARY_CSV = (
    CI / "conservation_priority/"
    "17m_primary_conservation_priority_100m.csv"
)

PRIMARY_GPKG = (
    CI / "conservation_priority/"
    "17m_primary_conservation_priority_100m.gpkg"
)

PRIMARY_QA = (
    REPORT / "17m_primary_conservation_priority_qa.json"
)

GI_CSV = (
    CI / "frozen_importance/"
    "17j_frozen_green_importance_100m.csv"
)

GI_GPKG = (
    CI / "frozen_importance/"
    "17j_frozen_green_importance_100m.gpkg"
)

GI_MANIFEST = (
    REPORT / "17j_green_importance_freeze_manifest.json"
)

WEIGHTS_CSV = (
    REPORT / "17h_critic_weights.csv"
)

CURRENT_CSV = (
    CI / "current_green/"
    "17b_current_green_condition_2025_100m.csv"
)

PERSISTENCE_CSV = (
    CI / "persistence/"
    "17c_green_persistence_2011_2024_100m.csv"
)

CONNECTIVITY_CSV = (
    CI / "connectivity/"
    "17d_green_connectivity_2025_100m.csv"
)

SUMMARY_OUT = REPORT / "17n_priority_sensitivity_scenarios.csv"
CELL_OUT = REPORT / "17n_priority_cell_rank_instability.csv"
QA_OUT = REPORT / "17n_priority_sensitivity_qa.json"

GI_FIELD = "green_importance_index_0_1"
UP_FIELD = "future_urbanization_pressure_score"
CP_FIELD = "primary_conservation_priority_0_1"

CRITERIA = [
    "G1_current_green_0_1",
    "G2_historical_persistence_0_1",
    "G3_local_green_context_0_1",
]

TOL = 1e-10


def sha256(path):

    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def load_unique(path, expected):

    df = pd.read_csv(path)

    if (
        "grid_id" not in df.columns
        or len(df) != expected
        or df.grid_id.isna().any()
        or not df.grid_id.is_unique
    ):
        raise ValueError(
            f"{path.name}: expected {expected:,} unique grid IDs"
        )

    return df.set_index("grid_id")


def numeric(frame, field):

    if field not in frame.columns:
        raise ValueError(f"Missing field: {field}")

    return pd.to_numeric(
        frame[field],
        errors="coerce",
    ).to_numpy(dtype=float)


def critic_weights(X, observation_weights=None):

    """
    Population-based CRITIC using Pearson correlation.

    If observation_weights is supplied, calculate weighted
    population SD and weighted Pearson correlations.
    """

    if observation_weights is None:

        observation_weights = np.ones(
            len(X),
            dtype=float,
        )

    p = np.asarray(
        observation_weights,
        dtype=float,
    )

    if (
        not np.isfinite(p).all()
        or (p <= 0).any()
        or p.sum() <= 0
    ):
        raise ValueError(
            "Invalid CRITIC observation weights"
        )

    p = p / p.sum()

    mean = np.sum(
        X * p[:, None],
        axis=0,
    )

    centred = X - mean

    covariance = (
        centred * p[:, None]
    ).T @ centred

    sd = np.sqrt(
        np.diag(covariance)
    )

    if (sd <= 1e-12).any():
        raise ValueError(
            "CRITIC criterion has insufficient variation"
        )

    correlation = covariance / np.outer(sd, sd)

    correlation = np.clip(
        correlation,
        -1.0,
        1.0,
    )

    np.fill_diagonal(
        correlation,
        1.0,
    )

    information = sd * np.sum(
        1 - correlation,
        axis=1,
    )

    if (
        not np.isfinite(information).all()
        or information.sum() <= 1e-12
    ):
        raise ValueError(
            "Cannot determine CRITIC information content"
        )

    return information / information.sum()


def ordinal_ranks(scores, grid_ids):

    """
    Descending rank with grid_id as deterministic tie breaker.
    Used only for comparing scenario rankings.
    """

    order = np.lexsort(
        (
            np.asarray(grid_ids).astype(str),
            -np.asarray(scores, dtype=float),
        )
    )

    ranks = np.empty(
        len(scores),
        dtype=int,
    )

    ranks[order] = np.arange(
        1,
        len(scores) + 1,
    )

    return ranks


def summary(x):

    values = np.asarray(x, dtype=float)

    return {
        "min": float(values.min()),
        "p25": float(np.quantile(values, 0.25)),
        "median": float(np.median(values)),
        "mean": float(values.mean()),
        "p75": float(np.quantile(values, 0.75)),
        "p95": float(np.quantile(values, 0.95)),
        "max": float(values.max()),
    }


def main():

    print("=" * 78)
    print(
        "SECTION 04 | STAGE 4.15 - "
        "CONSERVATION PRIORITY SENSITIVITY"
    )
    print("=" * 78)

    required = [
        PRIMARY_CSV,
        PRIMARY_GPKG,
        PRIMARY_QA,
        GI_CSV,
        GI_GPKG,
        GI_MANIFEST,
        WEIGHTS_CSV,
        CURRENT_CSV,
        PERSISTENCE_CSV,
        CONNECTIVITY_CSV,
    ]

    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)

    checks = {}

    def check(name, condition):

        checks[name] = bool(condition)

        print(
            f"[{'PASS' if checks[name] else 'FAIL'}] "
            f"{name}"
        )

    def gate(label):

        if not all(checks.values()):
            raise SystemExit(
                f"{label}: validation failed. "
                "No sensitivity reports exported."
            )

    # ==================================================
    # 1. CHECK FROZEN RESEARCH OUTPUTS
    # ==================================================

    print("\n[1] VERIFY FROZEN RESEARCH INPUTS")

    primary_qa = json.loads(
        PRIMARY_QA.read_text(encoding="utf-8")
    )

    manifest = json.loads(
        GI_MANIFEST.read_text(encoding="utf-8")
    )

    check(
        "stage_414_passed",
        primary_qa.get("status")
        == "PASS_MULTIPLICATIVE_CONSERVATION_PRIORITY_SENSITIVITY_PENDING",
    )

    check(
        "stage_411_frozen",
        manifest.get("status")
        == "PASS_GREEN_IMPORTANCE_FROZEN_BASELINE_SENSITIVITY_PENDING",
    )

    for kind, path in (
        ("csv", GI_CSV),
        ("gpkg", GI_GPKG),
    ):

        check(
            f"frozen_gi_{kind}_hash_valid",
            sha256(path)
            == manifest["frozen_outputs"][kind]["sha256"],
        )

    for kind, path in (
        ("csv", PRIMARY_CSV),
        ("gpkg", PRIMARY_GPKG),
    ):

        check(
            f"primary_cp_{kind}_hash_valid",
            sha256(path)
            == primary_qa["output_sha256"][kind],
        )

    check(
        "frozen_weights_hash_valid",
        sha256(WEIGHTS_CSV)
        == manifest["input_sha256"]["critic_weights_csv"],
    )

    gate("Frozen evidence")

    # ==================================================
    # 2. LOAD AND ALIGN OBSERVATIONS
    # ==================================================

    print("\n[2] LOAD COMMON-DOMAIN EVIDENCE")

    primary = load_unique(
        PRIMARY_CSV,
        9240,
    )

    gi = load_unique(
        GI_CSV,
        9240,
    )

    current = load_unique(
        CURRENT_CSV,
        9240,
    )

    persistence = load_unique(
        PERSISTENCE_CSV,
        9240,
    )

    connectivity = load_unique(
        CONNECTIVITY_CSV,
        9240,
    )

    ids = primary.index

    for label, frame in (
        ("gi", gi),
        ("current", current),
        ("persistence", persistence),
        ("connectivity", connectivity),
    ):

        check(
            f"{label}_grid_ids_match",
            set(frame.index) == set(ids),
        )

    gate("Grid identity")

    gi = gi.loc[ids]
    current = current.loc[ids]
    persistence = persistence.loc[ids]
    connectivity = connectivity.loc[ids]

    available_gi = (
        gi["green_importance_status"]
        .eq("GREEN_IMPORTANCE_CALCULATED")
        .to_numpy()
    )

    common = (
        primary["primary_priority_status"]
        .eq("FUTURE_CONSERVATION_PRIORITY_CALCULATED")
        .to_numpy()
    )

    check(
        "8727_gi_eligible_cells",
        int(available_gi.sum()) == 8727,
    )

    check(
        "4609_primary_cp_cells",
        int(common.sum()) == 4609,
    )

    check(
        "common_domain_subset_of_gi_population",
        bool(np.all(~common | available_gi)),
    )

    check(
        "primary_4631_unscored_cells_preserved",
        int(
            primary[CP_FIELD].isna().sum()
        ) == 4631,
    )

    gate("Population verification")

    X_all = gi.loc[
        available_gi,
        CRITERIA,
    ].apply(
        pd.to_numeric,
        errors="coerce",
    ).to_numpy(dtype=float)

    X_common = gi.loc[
        common,
        CRITERIA,
    ].apply(
        pd.to_numeric,
        errors="coerce",
    ).to_numpy(dtype=float)

    common_ids = ids[common].to_numpy()

    GI_common = numeric(
        gi.loc[common],
        GI_FIELD,
    )

    UP_common = numeric(
        primary.loc[common],
        UP_FIELD,
    )

    CP_primary = numeric(
        primary.loc[common],
        CP_FIELD,
    )

    check(
        "all_criteria_finite_0_1",
        X_all.shape == (8727, 3)
        and X_common.shape == (4609, 3)
        and np.isfinite(X_all).all()
        and np.all(
            (X_all >= -TOL)
            & (X_all <= 1 + TOL)
        ),
    )

    check(
        "primary_CP_formula_reproduced",
        np.isfinite(GI_common).all()
        and np.isfinite(UP_common).all()
        and np.isfinite(CP_primary).all()
        and np.allclose(
            GI_common * UP_common,
            CP_primary,
            rtol=0,
            atol=1e-12,
        ),
    )

    # ==================================================
    # 3. LOAD FROZEN CRITIC WEIGHTS
    # ==================================================

    print("\n[3] VERIFY PRIMARY CRITIC WEIGHTS")

    weight_df = pd.read_csv(
        WEIGHTS_CSV
    ).set_index("criterion")

    weight_df = weight_df.loc[
        ["G1", "G2", "G3"]
    ]

    weights = pd.to_numeric(
        weight_df["critic_weight"],
        errors="coerce",
    ).to_numpy(dtype=float)

    check(
        "critic_weights_sum_to_one",
        np.isfinite(weights).all()
        and (weights >= 0).all()
        and np.isclose(
            weights.sum(),
            1.0,
            atol=1e-12,
        ),
    )

    check(
        "frozen_gi_reproduced_from_criteria",
        np.allclose(
            X_common @ weights,
            GI_common,
            rtol=0,
            atol=1e-12,
        ),
    )

    check(
        "unweighted_critic_reproduces_frozen_weights",
        np.allclose(
            critic_weights(X_all),
            weights,
            rtol=0,
            atol=1e-10,
        ),
    )

    gate("Weight integrity")

    print(
        "Frozen G1/G2/G3 weights:",
        weights.tolist(),
    )

    # ==================================================
    # 4. PREPARE ALTERNATIVE CRITERIA
    # ==================================================

    print("\n[4] VERIFY ALTERNATIVE INDICATORS")

    alternatives = {
        "G1_CONFIRMED_GREEN_LOWER_BOUND": (
            current,
            "mapped_green_share_lower_bound",
            0,
        ),

        "G2_HISTORICAL_MEAN_GREEN": (
            persistence,
            "historical_mean_green_fraction",
            1,
        ),

        "G3_NEIGHBOUR_CONTEXT_500M": (
            connectivity,
            "green_neighbour_context_500m",
            2,
        ),
    }

    for name, (frame, field, j) in alternatives.items():

        values = numeric(
            frame.loc[available_gi],
            field,
        )

        check(
            f"{name}_complete_bounded",
            len(values) == 8727
            and np.isfinite(values).all()
            and np.all(
                (values >= -TOL)
                & (values <= 1 + TOL)
            ),
        )

    gate("Alternative indicators")

    # ==================================================
    # 5. DEFINE SENSITIVITY SCENARIOS
    # ==================================================

    print("\n[5] GENERATE SENSITIVITY SCENARIOS")

    scenario_scores = {}

    scenario_metadata = {}

    def add_scenario(name, score, method, gi_weights=None):

        score = np.asarray(
            score,
            dtype=float,
        )

        if (
            score.shape != (4609,)
            or not np.isfinite(score).all()
            or np.any(score < -TOL)
            or np.any(score > 1 + TOL)
        ):
            raise ValueError(
                f"Invalid sensitivity score for {name}"
            )

        if name in scenario_scores:
            raise ValueError(
                f"Duplicate scenario: {name}"
            )

        scenario_scores[name] = score

        scenario_metadata[name] = {
            "method": method,
            "gi_weights": (
                None
                if gi_weights is None
                else [
                    float(v)
                    for v in gi_weights
                ]
            ),
        }

    add_scenario(
        "APPROVED_MULTIPLICATIVE_BASELINE",
        CP_primary,
        "FROZEN_GI_X_FROZEN_UP",
        weights,
    )

    equal_weights = np.array(
        [1 / 3, 1 / 3, 1 / 3],
        dtype=float,
    )

    add_scenario(
        "EQUAL_GI_WEIGHTS",
        (X_common @ equal_weights) * UP_common,
        "EQUAL_DOMAIN_GI_WEIGHTING",
        equal_weights,
    )

    # Re-estimate CRITIC weights over ALL eligible
    # 8,727 GI cells using clipped KMC grid area.
    # Do not estimate them only on the 4,609 pressure cells.

    kmc_areas = numeric(
        gi.loc[available_gi],
        "kmc_overlap_area_m2",
    )

    area_weights = critic_weights(
        X_all,
        observation_weights=kmc_areas,
    )

    add_scenario(
        "AREA_WEIGHTED_CRITIC_GI",
        (X_common @ area_weights) * UP_common,
        "AREA_WEIGHTED_CRITIC",
        area_weights,
    )

    print(
        "Area-weighted CRITIC:",
        area_weights.tolist(),
    )

    # +/-10% RELATIVE perturbation for each criterion.
    # Other coefficients stay fixed before normalization.

    for j, code in enumerate(("G1", "G2", "G3")):

        for factor, suffix in (
            (0.90, "MINUS_10PCT"),
            (1.10, "PLUS_10PCT"),
        ):

            alternative_weights = weights.copy()

            alternative_weights[j] *= factor

            alternative_weights /= alternative_weights.sum()

            add_scenario(
                f"{code}_{suffix}_WEIGHT",
                (X_common @ alternative_weights)
                * UP_common,
                "NORMALIZED_RELATIVE_WEIGHT_PERTURBATION",
                alternative_weights,
            )

    # Hold original CRITIC weights fixed while
    # changing only ONE criterion at a time.
    #
    # This isolates indicator-definition sensitivity.
    # Re-estimating weights would test two changes at once.

    for name, (frame, field, j) in alternatives.items():

        alternative_X = X_common.copy()

        alternative_X[:, j] = numeric(
            frame.loc[common],
            field,
        )

        add_scenario(
            name,
            (alternative_X @ weights) * UP_common,
            "ONE_INDICATOR_SUBSTITUTION_FIXED_CRITIC",
            weights,
        )

    # ADDITIVE COUNTERFACTUAL ONLY.
    #
    # 50/50 is a hypothetical neutral tradeoff,
    # NOT a newly approved primary method.
    #
    # It must not replace the multiplicative CP.

    add_scenario(
        "ADDITIVE_EQUAL_TRADEOFF_COUNTERFACTUAL",
        0.5 * GI_common + 0.5 * UP_common,
        "HYPOTHETICAL_ADDITIVE_SENSITIVITY_ONLY",
        weights,
    )

    check(
        "expected_13_scenarios_including_baseline",
        len(scenario_scores) == 13,
    )

    check(
        "baseline_scenario_preserves_original_cp",
        np.array_equal(
            scenario_scores[
                "APPROVED_MULTIPLICATIVE_BASELINE"
            ],
            CP_primary,
        ),
    )

    check(
        "all_scenario_scores_bounded_and_complete",
        all(
            len(values) == 4609
            and np.isfinite(values).all()
            and np.all(values >= -TOL)
            and np.all(values <= 1 + TOL)
            for values in scenario_scores.values()
        ),
    )

    gate("Scenario construction")

    # ==================================================
    # 6. RANK-STABILITY DIAGNOSTICS
    # ==================================================

    print("\n[6] CALCULATE RANKING ROBUSTNESS")

    baseline_rank = ordinal_ranks(
        CP_primary,
        common_ids,
    )

    top_n = int(
        np.ceil(
            len(common_ids) * 0.10
        )
    )

    baseline_top = (
        baseline_rank <= top_n
    )

    records = []
    rank_arrays = {}

    print(
        "Fixed ranking domain:",
        len(common_ids),
    )

    print(
        "Top-decile diagnostic cell count:",
        top_n,
    )

    for name, scores in scenario_scores.items():

        scenario_rank = ordinal_ranks(
            scores,
            common_ids,
        )

        rank_arrays[name] = scenario_rank

        selected_top = (
            scenario_rank <= top_n
        )

        common_top_count = int(
            np.sum(
                baseline_top & selected_top
            )
        )

        union_top_count = int(
            np.sum(
                baseline_top | selected_top
            )
        )

        absolute_shift = np.abs(
            scenario_rank - baseline_rank
        )

        rho = float(
            spearmanr(
                CP_primary,
                scores,
            ).statistic
        )

        tau = float(
            kendalltau(
                CP_primary,
                scores,
            ).statistic
        )

        meta = scenario_metadata[name]

        w = meta["gi_weights"]

        row = {
            "scenario": name,
            "method": meta["method"],
            "compared_cells": len(common_ids),
            "top10_cell_count": top_n,
            "spearman_vs_primary": rho,
            "kendall_vs_primary": tau,
            "top10_overlap_count": common_top_count,
            "top10_retention_pct": (
                100 * common_top_count / top_n
            ),
            "top10_jaccard": (
                common_top_count / union_top_count
            ),
            "median_absolute_rank_shift": float(
                np.median(absolute_shift)
            ),
            "p95_absolute_rank_shift": float(
                np.quantile(
                    absolute_shift,
                    0.95,
                )
            ),
            "maximum_absolute_rank_shift": int(
                absolute_shift.max()
            ),
            "median_score": float(
                np.median(scores)
            ),
            "p95_score": float(
                np.quantile(scores, 0.95)
            ),
            "G1_weight_used": (
                np.nan if w is None else w[0]
            ),
            "G2_weight_used": (
                np.nan if w is None else w[1]
            ),
            "G3_weight_used": (
                np.nan if w is None else w[2]
            ),
        }

        records.append(row)

        print(
            f"{name}: "
            f"rho={rho:.4f}, "
            f"tau={tau:.4f}, "
            f"top10 retained="
            f"{row['top10_retention_pct']:.2f}%, "
            f"p95 rank shift="
            f"{row['p95_absolute_rank_shift']:.0f}"
        )

    results = pd.DataFrame(
        records
    )

    # ==================================================
    # 7. PER-CELL RANK INSTABILITY
    # ==================================================

    print("\n[7] CALCULATE CELL-LEVEL INSTABILITY")

    names = list(rank_arrays)

    rank_matrix = np.column_stack([
        rank_arrays[name]
        for name in names
    ])

    absolute_changes = np.abs(
        rank_matrix - baseline_rank[:, None]
    )

    worst_idx = np.argmax(
        absolute_changes,
        axis=1,
    )

    cell_output = pd.DataFrame({
        "grid_id": common_ids,
        "frozen_green_importance": GI_common,
        "frozen_future_pressure": UP_common,
        "approved_primary_CP": CP_primary,
        "primary_rank": baseline_rank,
        "minimum_scenario_rank": rank_matrix.min(axis=1),
        "maximum_scenario_rank": rank_matrix.max(axis=1),
        "maximum_absolute_rank_shift": (
            absolute_changes.max(axis=1)
        ),
        "median_absolute_rank_shift": (
            np.median(
                absolute_changes,
                axis=1,
            )
        ),
        "most_disruptive_scenario": [
            names[j]
            for j in worst_idx
        ],
        "primary_top10_diagnostic": baseline_top,
        "top10_in_all_scenarios": (
            (rank_matrix <= top_n).all(axis=1)
        ),
        "top10_in_any_scenario": (
            (rank_matrix <= top_n).any(axis=1)
        ),
    })

    check(
        "scenario_comparison_rows_13",
        len(results) == 13
        and results.scenario.is_unique,
    )

    check(
        "cell_instability_rows_4609",
        len(cell_output) == 4609
        and cell_output.grid_id.is_unique,
    )

    check(
        "all_robustness_correlations_finite",
        np.isfinite(
            results[
                [
                    "spearman_vs_primary",
                    "kendall_vs_primary",
                ]
            ].to_numpy(dtype=float)
        ).all(),
    )

    check(
        "all_primary_ranks_preserved",
        np.array_equal(
            cell_output.primary_rank.to_numpy(),
            baseline_rank,
        ),
    )

    check(
        "baseline_top10_exactly_selected",
        int(
            cell_output[
                "primary_top10_diagnostic"
            ].sum()
        ) == top_n,
    )

    check(
        "baseline_robustness_identity",
        np.isclose(
            results.iloc[0].spearman_vs_primary,
            1.0,
            atol=1e-12,
        )
        and np.isclose(
            results.iloc[0].kendall_vs_primary,
            1.0,
            atol=1e-12,
        )
        and results.iloc[0].top10_overlap_count == top_n
        and results.iloc[0].maximum_absolute_rank_shift == 0,
    )

    gate("Sensitivity calculations")

    # ==================================================
    # 8. EXPORT DIAGNOSTICS ONLY
    # ==================================================

    print("\n[8] EXPORT SENSITIVITY REPORTS")

    REPORT.mkdir(
        parents=True,
        exist_ok=True,
    )

    results.to_csv(
        SUMMARY_OUT,
        index=False,
    )

    cell_output.to_csv(
        CELL_OUT,
        index=False,
    )

    saved_summary = pd.read_csv(
        SUMMARY_OUT
    )

    saved_cells = pd.read_csv(
        CELL_OUT
    )

    check(
        "saved_scenario_report_valid",
        len(saved_summary) == 13
        and saved_summary.scenario.is_unique,
    )

    check(
        "saved_cell_report_valid",
        len(saved_cells) == 4609
        and saved_cells.grid_id.is_unique,
    )

    check(
        "primary_CP_unchanged_after_sensitivity",
        sha256(PRIMARY_CSV)
        == primary_qa["output_sha256"]["csv"]
        and sha256(PRIMARY_GPKG)
        == primary_qa["output_sha256"]["gpkg"],
    )

    check(
        "frozen_GI_unchanged_after_sensitivity",
        sha256(GI_CSV)
        == manifest["frozen_outputs"]["csv"]["sha256"]
        and sha256(GI_GPKG)
        == manifest["frozen_outputs"]["gpkg"]["sha256"],
    )

    gate("Export integrity")

    # ==================================================
    # 9. QA AND RESEARCH INTERPRETATION
    # ==================================================

    alternatives_only = results.loc[
        results["scenario"]
        != "APPROVED_MULTIPLICATIVE_BASELINE"
    ].copy()

    qa = {
        "section":
            "04_conservation_intelligence",

        "stage":
            "4.15",

        "status":
            "PASS_PRIORITY_SENSITIVITY_ANALYSIS",

        "primary_method_unchanged":
            "FROZEN_GI_MULTIPLIED_BY_FROZEN_UP",

        "primary_cp_modified":
            False,

        "frozen_gi_modified":
            False,

        "frozen_pressure_modified":
            False,

        "number_of_scenarios_including_baseline":
            len(results),

        "ranking_domain_cells":
            len(common_ids),

        "weight_estimation_population":
            int(available_gi.sum()),

        "top_decile_diagnostic_count":
            top_n,

        "top_decile_is_final_priority_class":
            False,

        "baseline_CRITIC_weights":
            {
                code: float(weights[i])
                for i, code in enumerate(
                    ["G1", "G2", "G3"]
                )
            },

        "area_weighted_CRITIC_weights":
            {
                code: float(area_weights[i])
                for i, code in enumerate(
                    ["G1", "G2", "G3"]
                )
            },

        "baseline_CP_distribution":
            summary(CP_primary),

        "alternative_spearman_range": [
            float(
                alternatives_only.spearman_vs_primary.min()
            ),
            float(
                alternatives_only.spearman_vs_primary.max()
            ),
        ],

        "alternative_top10_retention_pct_range": [
            float(
                alternatives_only.top10_retention_pct.min()
            ),
            float(
                alternatives_only.top10_retention_pct.max()
            ),
        ],

        "largest_p95_rank_shift_scenario": (
            alternatives_only.sort_values(
                "p95_absolute_rank_shift",
                ascending=False,
            ).iloc[0]["scenario"]
        ),

        "cells_in_primary_top_decile":
            int(baseline_top.sum()),

        "cells_in_top_decile_in_every_scenario":
            int(
                cell_output[
                    "top10_in_all_scenarios"
                ].sum()
            ),

        "cells_in_top_decile_in_any_scenario":
            int(
                cell_output[
                    "top10_in_any_scenario"
                ].sum()
            ),

        "checks": checks,

        "output_sha256": {
            "scenario_report":
                sha256(SUMMARY_OUT),

            "cell_rank_report":
                sha256(CELL_OUT),
        },

        "methodology_notes": [
            (
                "The approved primary conservation "
                "priority remains GI multiplied by "
                "model-derived future pressure."
            ),
            (
                "All comparisons use the same "
                "4609-cell common inference domain."
            ),
            (
                "CRITIC area sensitivity uses "
                "all 8727 GI-eligible observations."
            ),
            (
                "One-indicator substitutions retain "
                "original CRITIC weights to isolate "
                "indicator definition changes."
            ),
            (
                "The 50/50 additive case is a "
                "hypothetical comparator, not a "
                "new approved priority method."
            ),
            (
                "Top-decile comparison is a "
                "diagnostic, not a priority category."
            ),
            (
                "Historical Green-to-Built exposure "
                "remains supporting evidence only."
            ),
            (
                "Spearman and Kendall compare ranking "
                "agreement; neither validates "
                "ecological consequences."
            ),
            (
                "Spatial autocorrelation and shared "
                "source classification errors may "
                "affect apparent ranking stability."
            ),
        ],

        "final_priority_classes_assigned":
            False,

        "final_gis_qa_pending":
            True,

        "next_stage":
            "4.16 Final QA, documentation and GIS packaging",
    }

    QA_OUT.write_text(
        json.dumps(
            qa,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\n" + "=" * 78)
    print("STAGE 4.15: PASS_PRIORITY_SENSITIVITY_ANALYSIS")
    print("Scenarios compared:", len(results))
    print("Common-domain cells:", len(common_ids))
    print("Frozen GI or primary CP modified: NO")
    print("Scenario report:", SUMMARY_OUT)
    print("Cell instability report:", CELL_OUT)
    print("QA report:", QA_OUT)
    print("=" * 78)


if __name__ == "__main__":
    main()
