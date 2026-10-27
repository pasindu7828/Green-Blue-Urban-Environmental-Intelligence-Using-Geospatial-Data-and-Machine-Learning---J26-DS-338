
"""
J26-DS-338 | Section 04 - Conservation Intelligence
Stage 4.9: CRITIC objective weighting for independent Green Importance.

The approved methodology allows an independently justified weighting method.
CRITIC weights are based on criterion contrast (population standard deviation)
and criterion conflict (1 - Pearson correlation with other criteria).

Only the three Stage 4.8 benefit criteria are inputs:
    G1  2025 RAW_RF current green fraction
    G2  2011-2024 stabilized green-dominant-year share
    G3  2025 RAW_RF 250 m surrounding green context

Weight estimation population: all 8,727 currently green grid cells for which
all three indicators were standardized and marked READY_FOR_WEIGHTING.
Each eligible grid cell has equal weight in the CRITIC statistics. Edge-cell
area weighting is reserved as a sensitivity comparison.

NO Green Importance scores, conservation priority, historical/future threats,
AHP, SHAP, Development Suitability or zoning information are used/generated.

Output files (reports only):
    17h_critic_weights.csv
    17h_critic_weighting_qa.json
"""

from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = ROOT / "outputs/reports/conservation_intelligence"

SOURCE_CSV = (
    ROOT / "data/processed/conservation_intelligence/importance_criteria/"
    "17g_green_importance_criteria_standardized_100m.csv"
)

SOURCE_QA = REPORT_DIR / "17g_green_importance_criteria_qa.json"

WEIGHTS_CSV = REPORT_DIR / "17h_critic_weights.csv"
WEIGHTS_QA = REPORT_DIR / "17h_critic_weighting_qa.json"

CRITERIA = [
    (
        "G1",
        "G1_current_green_0_1",
        "2025 current green fraction",
    ),
    (
        "G2",
        "G2_historical_persistence_0_1",
        "2011-2024 green persistence",
    ),
    (
        "G3",
        "G3_local_green_context_0_1",
        "250 m surrounding green context",
    ),
]

TOL = 1e-9


def sha256_file(path):

    h = hashlib.sha256()

    with path.open("rb") as f:
        for block in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(block)

    return h.hexdigest()


def distribution(values):

    return {
        "min": float(np.min(values)),
        "p25": float(np.quantile(values, 0.25)),
        "median": float(np.median(values)),
        "p75": float(np.quantile(values, 0.75)),
        "max": float(np.max(values)),
    }


def main():

    print("=" * 76)
    print(
        "SECTION 04 | STAGE 4.9 - "
        "CRITIC GREEN IMPORTANCE WEIGHTING"
    )
    print("=" * 76)

    for path in (SOURCE_CSV, SOURCE_QA):
        if not path.is_file():
            raise FileNotFoundError(path)

    checks = {}

    def check(name, condition):

        checks[name] = bool(condition)

        print(
            f"[{'PASS' if checks[name] else 'FAIL'}] "
            f"{name}"
        )

    def require_pass(phase):

        if not all(checks.values()):
            raise SystemExit(
                f"{phase}: validation failed; "
                "no outputs generated"
            )

    # ==================================================
    # 1. VERIFY STAGE 4.8 SOURCE
    # ==================================================

    print("\n[1] VERIFY STAGE 4.8 DATA PRODUCT")

    stage_48 = json.loads(
        SOURCE_QA.read_text(encoding="utf-8")
    )

    check(
        "stage_48_standardized_candidates_passed",
        stage_48.get("status")
        == "PASS_STANDARDIZED_CANDIDATES_WEIGHTING_PENDING",
    )

    check(
        "stage_48_no_earlier_importance_score",
        stage_48.get("green_importance_calculated") is False,
    )

    check(
        "stage_48_no_threat_or_suitability_inputs",
        stage_48.get(
            "historical_loss_used_in_importance"
        ) is False
        and stage_48.get(
            "future_pressure_used_in_importance"
        ) is False
        and stage_48.get(
            "development_suitability_used_in_importance"
        ) is False,
    )

    require_pass("Upstream QA")

    df = pd.read_csv(SOURCE_CSV)

    columns = [
        item[1]
        for item in CRITERIA
    ]

    needed = {
        "grid_id",
        "criterion_availability_status",
        "current_green_condition_status",
        "source_current_product",
        *columns,
    }

    missing = sorted(
        needed - set(df.columns)
    )

    if missing:
        raise ValueError(
            "Stage 4.8 CSV lacks required columns: "
            f"{missing}"
        )

    check(
        "grid_9240_unique_ids",
        len(df) == 9240
        and df["grid_id"].notna().all()
        and df["grid_id"].is_unique,
    )

    check(
        "current_source_raw_rf",
        df["source_current_product"]
        .eq("RAW_RF")
        .all(),
    )

    status = (
        df["criterion_availability_status"]
        .astype(str)
    )

    current_status = (
        df["current_green_condition_status"]
        .astype(str)
    )

    ready = (
        status.eq("READY_FOR_WEIGHTING")
        .to_numpy(dtype=bool)
    )

    no_green = (
        status.eq("NO_CURRENT_GREEN_DETECTED")
        .to_numpy(dtype=bool)
    )

    unknown = (
        status.eq("UNKNOWN_2025_GREEN_CONDITION")
        .to_numpy(dtype=bool)
    )

    check(
        "ready_no_green_unknown_partition",
        int(ready.sum()) == 8727
        and int(no_green.sum()) == 468
        and int(unknown.sum()) == 45
        and bool(
            np.all(
                ready | no_green | unknown
            )
        ),
    )

    check(
        "status_matches_current_green_evidence",
        current_status[ready].tolist().count(
            "GREEN_DETECTED_IN_VALID_AREA"
        ) == int(ready.sum())
        and bool(
            np.all(
                current_status[no_green]
                == "NO_GREEN_DETECTED_IN_VALID_AREA"
            )
        )
        and bool(
            np.all(
                current_status[unknown]
                == "UNKNOWN_NO_VALID_LULC"
            )
        ),
    )

    # Only three selected criterion columns are read
    # into the matrix for objective weighting.

    source_values = (
        df[columns]
        .apply(pd.to_numeric, errors="coerce")
        .to_numpy(dtype=float)
    )

    X = source_values[ready]

    check(
        "exactly_three_proposed_criteria",
        X.shape == (8727, 3),
    )

    check(
        "all_ready_criteria_finite",
        bool(np.isfinite(X).all()),
    )

    check(
        "nonready_cells_have_no_criteria",
        bool(
            np.isnan(
                source_values[~ready]
            ).all()
        ),
    )

    check(
        "criteria_native_benefit_scale_0_to_1",
        bool(
            np.all(
                (X >= -TOL)
                & (X <= 1.0 + TOL)
            )
        ),
    )

    require_pass("Source integrity")

    print(
        "Grid cells retained in source:",
        f"{len(df):,}",
    )

    print(
        "Green cells used for weighting:",
        f"{len(X):,}",
    )

    print(
        "No current green detected:",
        int(no_green.sum()),
    )

    print(
        "Unknown current green:",
        int(unknown.sum()),
    )

    print(
        "Only G1, G2 and G3 enter CRITIC."
    )

    # ==================================================
    # 2. CALCULATE CRITIC CONTRAST AND CORRELATION
    # ==================================================

    print(
        "\n[2] COMPUTE CRITIC "
        "CONTRAST AND CORRELATION CONFLICT"
    )

    # These inputs are already standardized
    # 0-1 benefit criteria.
    #
    # No additional min-max or percentile
    # transformation is introduced.
    #
    # Population standard deviation (ddof=0)
    # is calculated across 8,727 cells.

    means = X.mean(axis=0)

    sd = X.std(
        axis=0,
        ddof=0,
    )

    check(
        "positive_criterion_variability",
        bool(
            np.all(sd > 1e-12)
        ),
    )

    require_pass("Criterion variability")

    # Standard CRITIC uses Pearson correlation.
    # Spearman is retained only as a diagnostic.

    pearson_raw = np.corrcoef(
        X,
        rowvar=False,
    )

    check(
        "pearson_matrix_finite",
        bool(
            np.isfinite(
                pearson_raw
            ).all()
        ),
    )

    check(
        "pearson_symmetric_unit_diagonal",
        bool(
            np.allclose(
                pearson_raw,
                pearson_raw.T,
                atol=1e-12,
            )
        )
        and bool(
            np.allclose(
                np.diag(pearson_raw),
                1.0,
                atol=1e-12,
            )
        ),
    )

    check(
        "pearson_correlations_in_bounds",
        bool(
            np.all(
                pearson_raw >= -1.0 - TOL
            )
        )
        and bool(
            np.all(
                pearson_raw <= 1.0 + TOL
            )
        ),
    )

    require_pass("Correlation validity")

    # Correct only insignificant numerical
    # correlation-bound deviations.

    pearson = np.clip(
        pearson_raw,
        -1.0,
        1.0,
    )

    np.fill_diagonal(
        pearson,
        1.0,
    )

    # ==================================================
    # CRITIC FORMULA
    #
    # Conflict_j = SUM_k(1 - Pearson_r_jk)
    #
    # Information_j = SD_j * Conflict_j
    #
    # Weight_j = Information_j / SUM_j(Information_j)
    # ==================================================

    conflict = np.sum(
        1.0 - pearson,
        axis=1,
    )

    information = (
        sd * conflict
    )

    total_information = float(
        information.sum()
    )

    check(
        "critic_conflict_nonnegative",
        bool(
            np.isfinite(
                conflict
            ).all()
        )
        and bool(
            np.all(
                conflict >= -TOL
            )
        ),
    )

    check(
        "critic_information_positive_finite",
        bool(
            np.isfinite(
                information
            ).all()
        )
        and bool(
            np.all(
                information >= 0.0
            )
        )
        and total_information > 1e-12,
    )

    require_pass("CRITIC information")

    weights = (
        information / total_information
    )

    check(
        "critic_weights_valid_and_sum_to_one",
        bool(
            np.isfinite(
                weights
            ).all()
        )
        and bool(
            np.all(
                weights >= 0.0
            )
        )
        and bool(
            np.all(
                weights <= 1.0
            )
        )
        and bool(
            np.isclose(
                weights.sum(),
                1.0,
                atol=1e-12,
            )
        ),
    )

    # Separate diagnostic:
    # Spearman does NOT determine CRITIC weights.

    spearman = np.asarray(
        spearmanr(X).statistic,
        dtype=float,
    )

    check(
        "spearman_diagnostic_finite",
        spearman.shape == (3, 3)
        and bool(
            np.isfinite(
                spearman
            ).all()
        ),
    )

    require_pass("Weight computation")

    # ==================================================
    # 3. DISPLAY CRITIC RESULTS
    # ==================================================

    print("\n[3] OBJECTIVE WEIGHTING RESULTS")

    for j, (
        code,
        field,
        description,
    ) in enumerate(CRITERIA):

        print(
            f"{code} | "
            f"SD={sd[j]:.6f} | "
            f"conflict={conflict[j]:.6f} | "
            f"information={information[j]:.6f} | "
            f"weight={weights[j]:.8f}"
        )

    names = [
        criterion[0]
        for criterion in CRITERIA
    ]

    print(
        "\nPearson correlation matrix "
        "(used by CRITIC):"
    )

    print(
        pd.DataFrame(
            pearson,
            index=names,
            columns=names,
        ).round(6).to_string()
    )

    print(
        "\nSpearman correlation matrix "
        "(diagnostic only):"
    )

    print(
        pd.DataFrame(
            spearman,
            index=names,
            columns=names,
        ).round(6).to_string()
    )

    print(
        f"\nWeight sum: {weights.sum():.12f}"
    )

    # ==================================================
    # 4. VERIFY INDEPENDENCE
    # ==================================================

    print("\n[4] VERIFY OUTPUT CONTRACT")

    check(
        "no_importance_index_in_input",
        not any(
            "importance_index" in col.lower()
            for col in df.columns
        ),
    )

    forbidden = [
        "pressure",
        "shap",
        "suitability",
        "zoning",
        "historical_loss",
        "historical_g2b",
    ]

    check(
        "source_has_no_threat_pressure_or_suitability_columns",
        not any(
            term in col.lower()
            for col in df.columns
            for term in forbidden
        ),
    )

    require_pass("Independence")

    # ==================================================
    # 5. BUILD THREE-CRITERION WEIGHT REPORT
    # ==================================================

    rows = []

    for j, (
        code,
        field,
        description,
    ) in enumerate(CRITERIA):

        rows.append({
            "criterion":
                code,

            "source_field":
                field,

            "description":
                description,

            "direction":
                "BENEFIT",

            "standardization":
                "IDENTITY_0_1",

            "observed_cells":
                int(X.shape[0]),

            "mean":
                float(means[j]),

            "population_sd":
                float(sd[j]),

            "contrast_intensity":
                float(sd[j]),

            "correlation_conflict":
                float(conflict[j]),

            "information_content":
                float(information[j]),

            "critic_weight":
                float(weights[j]),

            "percent_at_one":
                float(
                    100.0
                    * np.isclose(
                        X[:, j],
                        1.0,
                        atol=1e-9,
                        rtol=0.0,
                    ).mean()
                ),
        })

    result = pd.DataFrame(rows)

    check(
        "weight_export_three_distinct_criteria",
        len(result) == 3
        and result["criterion"].is_unique,
    )

    require_pass("Final weight table")

    # Hash enables Stage 4.10 to verify that the
    # weighting input has not changed.

    source_hash = sha256_file(
        SOURCE_CSV
    )

    # ==================================================
    # 6. BUILD METHODOLOGY AND QA REPORT
    # ==================================================

    qa = {
        "section":
            "04_conservation_intelligence",

        "stage":
            "4.9",

        "status":
            "PASS_CRITIC_WEIGHTING_INDEX_PENDING",

        "method":
            "CRITIC_OBJECTIVE_WEIGHTING",

        "methodology_decision": (
            "Researcher approved CRITIC for "
            "Green Importance because the approved "
            "Section 04 methodology prescribed no "
            "mandatory weighting method. "
            "Criterion variability and Pearson "
            "correlation conflict determine weights "
            "without fabricated expert judgments."
        ),

        "input_csv_relative_path": (
            str(
                SOURCE_CSV.relative_to(ROOT)
            ).replace("\\", "/")
        ),

        "input_csv_sha256":
            source_hash,

        "stage_48_qa_status":
            stage_48["status"],

        "candidate_selection_previously_marked_provisional":
            bool(
                stage_48.get(
                    "candidate_selection_provisional",
                    True,
                )
            ),

        "weighting_population":
            "READY_FOR_WEIGHTING current-green cells",

        "weighting_observations":
            int(X.shape[0]),

        "source_grid_cells":
            int(len(df)),

        "population_weighting":
            "EQUAL_PER_GRID_CELL_NOT_AREA_WEIGHTED",

        "std_convention":
            "POPULATION_DDOF_0",

        "correlation_for_CRITIC":
            "PEARSON",

        "diagnostic_correlation_only":
            "SPEARMAN",

        "standardization":
            "NATIVE_BENEFIT_PROPORTION_IDENTITY_0_1",

        "formula": (
            "C_j = population_sd(G_j) * "
            "SUM_k(1 - Pearson_r_jk); "
            "w_j = C_j / SUM_j(C_j)"
        ),

        "criteria":
            rows,

        "pearson_correlation": {
            code_i: {
                code_j: float(
                    pearson[i, j]
                )
                for j, (
                    code_j, _, _
                ) in enumerate(CRITERIA)
            }
            for i, (
                code_i, _, _
            ) in enumerate(CRITERIA)
        },

        "spearman_diagnostic": {
            code_i: {
                code_j: float(
                    spearman[i, j]
                )
                for j, (
                    code_j, _, _
                ) in enumerate(CRITERIA)
            }
            for i, (
                code_i, _, _
            ) in enumerate(CRITERIA)
        },

        "criterion_distributions": {
            code: distribution(
                X[:, i]
            )
            for i, (
                code, _, _
            ) in enumerate(CRITERIA)
        },

        "weight_sum":
            float(weights.sum()),

        "checks":
            checks,

        "green_importance_index_calculated":
            False,

        "green_importance_frozen":
            False,

        "historical_loss_used_in_weighting":
            False,

        "future_pressure_used_in_weighting":
            False,

        "development_suitability_ahp_used":
            False,

        "pending": [
            (
                "Review CRITIC weights and saturation "
                "of persistence before index calculation"
            ),
            (
                "Stage 4.10: construct and validate "
                "Green Importance Index"
            ),
            (
                "Stage 4.11: freeze Green Importance"
            ),
            (
                "Stage 4.15: compare equal-contribution, "
                "alternative indicator, area-weighted "
                "CRITIC and perturbation scenarios"
            ),
        ],

        "limitations": [
            (
                "CRITIC derives statistical contrast "
                "and conflict, not expert ecological values"
            ),
            (
                "Spatial grid observations are "
                "autocorrelated; weights are descriptive"
            ),
            (
                "Each KMC grid cell contributes equally "
                "regardless of clipped area"
            ),
            (
                "Historical green-dominant persistence "
                "has many values at one"
            ),
            (
                "No importance index, conservation "
                "priority or threat score is created"
            ),
        ],
    }

    # ==================================================
    # 7. EXPORT ONLY THE CRITIC WEIGHTS
    # ==================================================

    print("\n[5] EXPORT OBJECTIVE WEIGHT REPORTS")

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    result.to_csv(
        WEIGHTS_CSV,
        index=False,
    )

    WEIGHTS_QA.write_text(
        json.dumps(
            qa,
            indent=2,
        ),
        encoding="utf-8",
    )

    saved = pd.read_csv(
        WEIGHTS_CSV
    )

    check(
        "saved_weights_reproduced",
        len(saved) == 3
        and np.allclose(
            saved["critic_weight"].to_numpy(),
            weights,
            atol=1e-12,
        )
        and np.isclose(
            saved["critic_weight"].sum(),
            1.0,
            atol=1e-12,
        ),
    )

    if not all(checks.values()):
        raise SystemExit(
            "Saved weight evidence "
            "did not pass verification"
        )

    qa["checks"] = checks

    WEIGHTS_QA.write_text(
        json.dumps(
            qa,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("Weights CSV:", WEIGHTS_CSV)
    print("CRITIC QA JSON:", WEIGHTS_QA)

    print("\n" + "=" * 76)
    print(
        "STAGE 4.9: "
        "PASS_CRITIC_WEIGHTING_INDEX_PENDING"
    )
    print("Green Importance Index: NOT CALCULATED")
    print("Green Importance freeze: NOT PERFORMED")
    print("Historical/future threat used: NO")
    print("=" * 76)


if __name__ == "__main__":
    main()
