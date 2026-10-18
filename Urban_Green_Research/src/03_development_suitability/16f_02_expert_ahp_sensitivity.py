
"""
J26-DS-338 | Section 03 | Step 16F-02

POST-AHP DEVELOPMENT SUITABILITY SENSITIVITY ANALYSIS

Baseline:
    Engineer AHP weights:
    C1 = 0.20, C2 = 0.60, C4 = 0.20

Method:
    12 hypothetical weight-transfer scenarios:
    5 and 10 percentage points between each ordered
    pair of criteria, plus the original AHP baseline.

Evaluation:
    - Spearman rank correlation
    - Absolute suitability score changes
    - Grid-level ranking stability
    - Top 10% and top 20% retention
    - Area-weighted score changes

IMPORTANT:
    Hypothetical weights are sensitivity tests only.
    They are NOT new expert judgments.

    No changes to frozen input files.
    No changes to Section 04.
    No classification thresholds assigned.
"""

from pathlib import Path
from tempfile import TemporaryDirectory
import hashlib
import json

import numpy as np
import pandas as pd


# ============================================================
# 1. PROJECT PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

BASE = (
    ROOT
    / "data/processed/development_suitability"
    / "final_expert_ahp"
)

REPORT = ROOT / "outputs/reports"

SCORE_CSV = (
    BASE
    / "16e_02_expert_ahp_weighted_suitability_100m.csv"
)

SCORE_GPKG = (
    BASE
    / "16e_02_expert_ahp_weighted_suitability_100m.gpkg"
)

SCORE_QA = (
    REPORT
    / "16e_02_expert_ahp_weighted_qa.json"
)

EXPERT_QA = (
    REPORT
    / "16d_e_expert_ahp_consistency_qa.json"
)

EXPERT_WEIGHTS = (
    REPORT
    / "16d_e_expert_ahp_weights.csv"
)

OUT_SUMMARY = (
    BASE
    / "16f_02_sensitivity_scenario_summary.csv"
)

OUT_GRID = (
    BASE
    / "16f_02_sensitivity_grid_stability.csv"
)

OUT_QA = (
    REPORT
    / "16f_02_expert_ahp_sensitivity_qa.json"
)


# ============================================================
# 2. FROZEN RESEARCH CONFIGURATION
# ============================================================

CRITERIA = ("C1", "C2", "C4")

SCORE_COLUMNS = (
    "c1_road_access_score",
    "c2_road_context_score",
    "c4_slope_score",
)

BASELINE_FIELD = (
    "development_suitability_score"
)

EXPECTED_N = 9240

# These are hypothetical percentage-point transfers.
# They DO NOT replace the engineer's AHP judgments.

DELTAS = (0.05, 0.10)


# ============================================================
# 3. HELPER FUNCTIONS
# ============================================================

def sha256(path):

    digest = hashlib.sha256()

    with path.open("rb") as file:

        for block in iter(
            lambda: file.read(1024 * 1024),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def require(condition, description):

    passed = bool(condition)

    print(
        f"[{'PASS' if passed else 'FAIL'}] "
        f"{description}"
    )

    if not passed:

        raise ValueError(
            f"QA FAILED: {description}"
        )


def finite_column(frame, name):

    if name not in frame.columns:

        raise ValueError(
            f"Missing field: {name}"
        )

    values = pd.to_numeric(
        frame[name],
        errors="coerce",
    ).to_numpy(dtype=float)

    require(
        np.isfinite(values).all(),
        f"{name}: finite values",
    )

    return values


def deterministic_ranks(scores, grid_ids):

    """
    Rank 1 = highest development suitability.

    Equal scores are ordered by grid_id.
    This makes top-group selection reproducible.
    """

    order = np.lexsort(
        (grid_ids, -scores)
    )

    ranks = np.empty(
        len(scores),
        dtype=np.int32,
    )

    ranks[order] = np.arange(
        1,
        len(scores) + 1,
    )

    return ranks


def build_scenarios(base_weights):

    """
    One original AHP baseline +
    12 hypothetical weight-transfer scenarios.

    Six directional transfers:
        C1 -> C2
        C1 -> C4
        C2 -> C1
        C2 -> C4
        C4 -> C1
        C4 -> C2

    Two magnitudes:
        5 percentage points
        10 percentage points
    """

    scenarios = []

    scenarios.append({
        "scenario_id": "BASE_EXPERT_AHP",
        "scenario_type": "BASELINE",
        "transfer_from": "",
        "transfer_to": "",
        "transfer_pp": 0.0,
        "weights": base_weights.copy(),
    })

    for source in CRITERIA:

        for target in CRITERIA:

            if source == target:
                continue

            source_index = CRITERIA.index(source)
            target_index = CRITERIA.index(target)

            for delta in DELTAS:

                changed = base_weights.copy()

                changed[source_index] -= delta
                changed[target_index] += delta

                require(
                    np.all(changed >= -1e-12)
                    and np.isclose(
                        changed.sum(),
                        1.0,
                        atol=1e-12,
                    ),
                    (
                        f"Valid weight transfer: "
                        f"{source} to {target}, "
                        f"{delta:.2f}"
                    ),
                )

                scenarios.append({
                    "scenario_id": (
                        f"TRANSFER_{int(round(delta * 100)):02d}"
                        f"PP_{source}_TO_{target}"
                    ),
                    "scenario_type": (
                        "HYPOTHETICAL_WEIGHT_STRESS_TEST"
                    ),
                    "transfer_from": source,
                    "transfer_to": target,
                    "transfer_pp": delta * 100,
                    "weights": changed,
                })

    require(
        len(scenarios) == 13,
        "Exactly 13 sensitivity scenarios",
    )

    require(
        len({
            s["scenario_id"]
            for s in scenarios
        }) == 13,
        "Scenario IDs unique",
    )

    return scenarios


# ============================================================
# 4. MAIN WORKFLOW
# ============================================================

def main():

    print("=" * 78)

    print(
        "SECTION 03 | STEP 16F-02 "
        "POST-AHP SENSITIVITY ANALYSIS"
    )

    print("=" * 78)

    inputs = [
        SCORE_CSV,
        SCORE_GPKG,
        SCORE_QA,
        EXPERT_QA,
        EXPERT_WEIGHTS,
    ]

    outputs = [
        OUT_SUMMARY,
        OUT_GRID,
        OUT_QA,
    ]

    for path in inputs:

        if not path.is_file():

            raise FileNotFoundError(path)

    for path in outputs:

        if path.exists():

            raise FileExistsError(
                f"Refusing to overwrite: {path}"
            )

    input_hashes = {
        path.relative_to(ROOT).as_posix(): sha256(path)
        for path in inputs
    }

    # ========================================================
    # STEP 1 - VERIFY FINAL SUITABILITY AND EXPERT AHP
    # ========================================================

    print("\n[1] VERIFY FROZEN RESEARCH INPUTS")

    score_qa = json.loads(
        SCORE_QA.read_text(
            encoding="utf-8"
        )
    )

    expert_qa = json.loads(
        EXPERT_QA.read_text(
            encoding="utf-8"
        )
    )

    require(
        score_qa.get("status")
        == (
            "PASS_EXPERT_AHP_WEIGHTED_SUITABILITY_"
            "SENSITIVITY_PENDING"
        ),
        "Final AHP suitability calculation passed",
    )

    require(
        score_qa.get("grid_cells") == EXPECTED_N,
        "9,240 final suitability grid cells",
    )

    require(
        score_qa.get("crs") == "EPSG:32644",
        "Final suitability GIS CRS",
    )

    require(
        score_qa.get("outputs", {}).get(
            "csv_sha256"
        ) == sha256(SCORE_CSV),
        "Final suitability CSV hash verified",
    )

    require(
        score_qa.get("outputs", {}).get(
            "gpkg_sha256"
        ) == sha256(SCORE_GPKG),
        "Final suitability GIS hash verified",
    )

    require(
        expert_qa.get("scenario")
        == "BASE_3_CRITERIA",
        "Correct expert AHP scenario",
    )

    require(
        expert_qa.get("matrix_criterion_order")
        == list(CRITERIA),
        "Correct AHP criterion order",
    )

    require(
        expert_qa.get("consistency_acceptable")
        is True
        and float(
            expert_qa.get(
                "consistency_ratio",
                1.0,
            )
        ) < 0.10,
        "AHP consistency accepted",
    )

    # ========================================================
    # STEP 2 - VERIFY EXPERT WEIGHTS
    # ========================================================

    print("\n[2] VERIFY EXPERT AHP WEIGHTS")

    weight_table = pd.read_csv(
        EXPERT_WEIGHTS
    )

    require(
        len(weight_table) == 3
        and weight_table["criterion"].is_unique
        and set(weight_table["criterion"])
        == set(CRITERIA),
        "Three unique criterion weights",
    )

    weight_table = (
        weight_table
        .set_index("criterion")
        .loc[list(CRITERIA)]
    )

    weights = pd.to_numeric(
        weight_table["expert_ahp_weight"],
        errors="coerce",
    ).to_numpy(dtype=float)

    require(
        np.isfinite(weights).all()
        and np.allclose(
            weights,
            [0.2, 0.6, 0.2],
            atol=1e-12,
            rtol=0,
        ),
        "Expert weights C1=.20 C2=.60 C4=.20",
    )

    require(
        np.isclose(
            weights.sum(),
            1.0,
            atol=1e-12,
        ),
        "Expert weights sum to 1",
    )

    require(
        all(
            np.isclose(
                float(
                    expert_qa["weights"][criterion]
                ),
                weights[i],
                atol=1e-12,
            )
            for i, criterion in enumerate(CRITERIA)
        ),
        "Expert weights agree with AHP QA",
    )

    recorded_hashes = score_qa.get(
        "input_sha256",
        {},
    )

    for path in [
        EXPERT_QA,
        EXPERT_WEIGHTS,
    ]:

        key = path.relative_to(ROOT).as_posix()

        require(
            recorded_hashes.get(key)
            == sha256(path),
            f"Expert source provenance: {path.name}",
        )

    print(
        "Baseline expert weights:",
        weights.tolist(),
    )

    # ========================================================
    # STEP 3 - LOAD AND VALIDATE GRID SCORES
    # ========================================================

    print("\n[3] VALIDATE FINAL SUITABILITY DATA")

    frame = pd.read_csv(
        SCORE_CSV
    )

    require(
        len(frame) == EXPECTED_N,
        "9,240 suitability rows",
    )

    require(
        frame["grid_id"].notna().all()
        and frame["grid_id"].is_unique,
        "Unique nonmissing grid IDs",
    )

    grid_ids = (
        frame["grid_id"]
        .astype(str)
        .to_numpy()
    )

    area = finite_column(
        frame,
        "kmc_overlap_area_m2",
    )

    require(
        np.all(area > 0),
        "Positive grid intersection areas",
    )

    require(
        np.isclose(
            area.sum() / 1e6,
            88.82921581108494,
            atol=0.001,
            rtol=0,
        ),
        "KMC study area verified",
    )

    X = np.column_stack([
        finite_column(frame, column)
        for column in SCORE_COLUMNS
    ])

    require(
        X.shape == (9240, 3),
        "Standardized criterion matrix 9240 x 3",
    )

    require(
        np.all(X >= 0)
        and np.all(X <= 1),
        "Standardized scores bounded [0,1]",
    )

    baseline = finite_column(
        frame,
        BASELINE_FIELD,
    )

    require(
        np.all(baseline >= 0)
        and np.all(baseline <= 1),
        "Baseline suitability bounded [0,1]",
    )

    require(
        np.allclose(
            X @ weights,
            baseline,
            atol=1e-12,
            rtol=0,
        ),
        "Baseline suitability reproduced exactly",
    )

    # ========================================================
    # STEP 4 - BASELINE RANKS AND TOP GROUPS
    # ========================================================

    print("\n[4] PREPARE BASELINE RANKINGS")

    n = len(frame)

    top10_size = int(
        np.ceil(n * 0.10)
    )

    top20_size = int(
        np.ceil(n * 0.20)
    )

    baseline_rank = deterministic_ranks(
        baseline,
        grid_ids,
    )

    baseline_top10 = (
        baseline_rank <= top10_size
    )

    baseline_top20 = (
        baseline_rank <= top20_size
    )

    require(
        baseline_top10.sum() == top10_size,
        "Baseline top 10% has correct size",
    )

    require(
        baseline_top20.sum() == top20_size,
        "Baseline top 20% has correct size",
    )

    print(
        "Top 10% cells:", top10_size
    )

    print(
        "Top 20% cells:", top20_size
    )

    print(
        "NOTE: Top groups are analytical ranking "
        "sets, not final suitability classes."
    )

    # ========================================================
    # STEP 5 - GENERATE HYPOTHETICAL SCENARIOS
    # ========================================================

    print("\n[5] BUILD SENSITIVITY SCENARIOS")

    scenarios = build_scenarios(
        weights
    )

    # ========================================================
    # STEP 6 - COMPUTE SENSITIVITY METRICS
    # ========================================================

    print("\n[6] EVALUATE WEIGHT SENSITIVITY")

    records = []

    min_scores = baseline.copy()
    max_scores = baseline.copy()

    max_score_change = np.zeros(n)
    total_score_change = np.zeros(n)

    max_rank_shift = np.zeros(
        n,
        dtype=np.int32,
    )

    total_rank_shift = np.zeros(n)

    top10_count = np.zeros(
        n,
        dtype=np.int16,
    )

    top20_count = np.zeros(
        n,
        dtype=np.int16,
    )

    for scenario in scenarios:

        scenario_id = scenario["scenario_id"]
        scenario_weights = scenario["weights"]

        scenario_scores = (
            X @ scenario_weights
        )

        require(
            np.isfinite(scenario_scores).all()
            and np.all(
                scenario_scores >= -1e-12
            )
            and np.all(
                scenario_scores <= 1 + 1e-12
            ),
            f"Valid scores: {scenario_id}",
        )

        ranks = deterministic_ranks(
            scenario_scores,
            grid_ids,
        )

        top10 = (
            ranks <= top10_size
        )

        top20 = (
            ranks <= top20_size
        )

        top10_count += top10.astype(
            np.int16
        )

        top20_count += top20.astype(
            np.int16
        )

        score_change = np.abs(
            scenario_scores - baseline
        )

        rank_shift = np.abs(
            ranks - baseline_rank
        )

        min_scores = np.minimum(
            min_scores,
            scenario_scores,
        )

        max_scores = np.maximum(
            max_scores,
            scenario_scores,
        )

        max_score_change = np.maximum(
            max_score_change,
            score_change,
        )

        total_score_change += score_change

        max_rank_shift = np.maximum(
            max_rank_shift,
            rank_shift,
        )

        total_rank_shift += rank_shift

        correlation = float(
            pd.Series(baseline).corr(
                pd.Series(scenario_scores),
                method="spearman",
            )
        )

        retained10 = int(
            np.count_nonzero(
                baseline_top10 & top10
            )
        )

        retained20 = int(
            np.count_nonzero(
                baseline_top20 & top20
            )
        )

        records.append({
            "scenario_id": scenario_id,
            "scenario_type": scenario[
                "scenario_type"
            ],
            "transfer_from": scenario[
                "transfer_from"
            ],
            "transfer_to": scenario[
                "transfer_to"
            ],
            "transferred_weight_percentage_points":
                scenario["transfer_pp"],
            "C1_weight": float(
                scenario_weights[0]
            ),
            "C2_weight": float(
                scenario_weights[1]
            ),
            "C4_weight": float(
                scenario_weights[2]
            ),
            "spearman_score_correlation":
                correlation,
            "mean_absolute_score_change":
                float(score_change.mean()),
            "area_weighted_mean_absolute_score_change":
                float(
                    np.average(
                        score_change,
                        weights=area,
                    )
                ),
            "p95_absolute_score_change":
                float(
                    np.quantile(
                        score_change,
                        0.95,
                    )
                ),
            "maximum_absolute_score_change":
                float(score_change.max()),
            "median_absolute_rank_shift":
                float(
                    np.median(rank_shift)
                ),
            "p95_absolute_rank_shift":
                float(
                    np.quantile(
                        rank_shift,
                        0.95,
                    )
                ),
            "maximum_absolute_rank_shift":
                int(rank_shift.max()),
            "baseline_top10_count_retained":
                retained10,
            "baseline_top10_retention_fraction":
                float(
                    retained10 / top10_size
                ),
            "baseline_top20_count_retained":
                retained20,
            "baseline_top20_retention_fraction":
                float(
                    retained20 / top20_size
                ),
        })

        print(
            f"{scenario_id}: "
            f"Spearman={correlation:.6f}, "
            f"Top10 retained={retained10}/{top10_size}, "
            f"Max change={score_change.max():.6f}"
        )

    # ========================================================
    # STEP 7 - GRID-LEVEL ROBUSTNESS
    # ========================================================

    print("\n[7] COMPUTE GRID-LEVEL STABILITY")

    scenario_count = len(scenarios)

    grid_stability = pd.DataFrame({

        "grid_id": frame["grid_id"],

        "kmc_overlap_area_m2": area,

        "baseline_development_suitability_score":
            baseline,

        "baseline_suitability_rank":
            baseline_rank,

        "baseline_top10pct":
            baseline_top10,

        "baseline_top20pct":
            baseline_top20,

        "scenario_count":
            scenario_count,

        "top10pct_scenario_count":
            top10_count,

        "top10pct_scenario_fraction":
            top10_count / scenario_count,

        "top20pct_scenario_count":
            top20_count,

        "top20pct_scenario_fraction":
            top20_count / scenario_count,

        "top10pct_in_all_scenarios":
            top10_count == scenario_count,

        "top20pct_in_all_scenarios":
            top20_count == scenario_count,

        "minimum_scenario_score":
            min_scores,

        "maximum_scenario_score":
            max_scores,

        "score_range_across_scenarios":
            max_scores - min_scores,

        "maximum_absolute_score_change":
            max_score_change,

        "mean_absolute_score_change":
            total_score_change / scenario_count,

        "maximum_absolute_rank_shift":
            max_rank_shift,

        "mean_absolute_rank_shift":
            total_rank_shift / scenario_count,

    })

    summary = pd.DataFrame(
        records
    )

    # ========================================================
    # STEP 8 - VALIDATE SENSITIVITY RESULTS
    # ========================================================

    print("\n[8] VALIDATE SENSITIVITY RESULTS")

    require(
        len(summary) == 13
        and summary["scenario_id"].is_unique,
        "13 unique scenario records",
    )

    require(
        len(grid_stability) == EXPECTED_N
        and grid_stability["grid_id"].is_unique,
        "9,240 grid-level stability records",
    )

    require(
        np.all(
            (top10_count >= 0)
            & (top10_count <= scenario_count)
        ),
        "Top 10% membership counts valid",
    )

    require(
        np.all(
            (top20_count >= 0)
            & (top20_count <= scenario_count)
        ),
        "Top 20% membership counts valid",
    )

    require(
        int(top10_count.sum())
        == scenario_count * top10_size,
        "Top 10% membership conservation",
    )

    require(
        int(top20_count.sum())
        == scenario_count * top20_size,
        "Top 20% membership conservation",
    )

    require(
        np.all(min_scores <= baseline + 1e-12)
        and np.all(max_scores >= baseline - 1e-12),
        "Sensitivity intervals contain baseline",
    )

    require(
        summary.iloc[0]["scenario_id"]
        == "BASE_EXPERT_AHP"
        and np.isclose(
            summary.iloc[0][
                "mean_absolute_score_change"
            ],
            0.0,
            atol=1e-12,
        ),
        "Unchanged expert baseline reproduced",
    )

    # ========================================================
    # STEP 9 - EXPORT REPORTS
    # ========================================================

    print("\n[9] EXPORT AUDITABLE SENSITIVITY FILES")

    BASE.mkdir(
        parents=True,
        exist_ok=True,
    )

    with TemporaryDirectory(
        prefix="16f_02_build_",
        dir=BASE,
    ) as folder:

        folder = Path(folder)

        temporary_summary = (
            folder / OUT_SUMMARY.name
        )

        temporary_grid = (
            folder / OUT_GRID.name
        )

        summary.to_csv(
            temporary_summary,
            index=False,
        )

        grid_stability.to_csv(
            temporary_grid,
            index=False,
        )

        saved_summary = pd.read_csv(
            temporary_summary
        )

        saved_grid = pd.read_csv(
            temporary_grid
        )

        require(
            len(saved_summary) == 13
            and saved_summary["scenario_id"].is_unique,
            "Exported scenario summary verified",
        )

        require(
            len(saved_grid) == EXPECTED_N
            and saved_grid["grid_id"].is_unique,
            "Exported grid stability verified",
        )

        require(
            np.allclose(
                saved_grid[
                    "baseline_development_suitability_score"
                ].to_numpy(dtype=float),
                baseline,
                atol=1e-12,
                rtol=0,
            ),
            "Exported baseline scores unchanged",
        )

        for output in outputs:

            if output.exists():

                raise FileExistsError(
                    f"Unexpected existing output: {output}"
                )

        temporary_summary.replace(
            OUT_SUMMARY
        )

        temporary_grid.replace(
            OUT_GRID
        )

    # ========================================================
    # STEP 10 - VERIFY INPUT INTEGRITY
    # ========================================================

    print("\n[10] VERIFY ORIGINAL INPUTS UNCHANGED")

    for path in inputs:

        relative = (
            path.relative_to(ROOT).as_posix()
        )

        require(
            sha256(path) == input_hashes[relative],
            f"Source unchanged: {path.name}",
        )

    # ========================================================
    # STEP 11 - FINAL SENSITIVITY QA REPORT
    # ========================================================

    print("\n[11] GENERATE SENSITIVITY QA REPORT")

    stress = summary.loc[
        summary["scenario_type"] != "BASELINE"
    ]

    diagnostics = {

        "worst_scenario_spearman_correlation":
            float(
                stress[
                    "spearman_score_correlation"
                ].min()
            ),

        "smallest_top_decile_retention_fraction":
            float(
                stress[
                    "baseline_top10_retention_fraction"
                ].min()
            ),

        "smallest_top_quintile_retention_fraction":
            float(
                stress[
                    "baseline_top20_retention_fraction"
                ].min()
            ),

        "largest_scenario_max_abs_score_change":
            float(
                stress[
                    "maximum_absolute_score_change"
                ].max()
            ),

        "largest_scenario_p95_abs_rank_shift":
            float(
                stress[
                    "p95_absolute_rank_shift"
                ].max()
            ),

        "grids_always_in_top_decile":
            int(
                grid_stability[
                    "top10pct_in_all_scenarios"
                ].sum()
            ),

        "grids_always_in_top_quintile":
            int(
                grid_stability[
                    "top20pct_in_all_scenarios"
                ].sum()
            ),
    }

    qa = {

        "section":
            "03_development_suitability",

        "stage":
            "16f_02",

        "status": (
            "PASS_POST_AHP_SENSITIVITY_WITH_"
            "DOCUMENTED_STRESS_TESTS_"
            "FINAL_RELEASE_PENDING"
        ),

        "baseline": (
            "RESEARCHER_TRANSCRIBED_ENGINEER_AHP"
        ),

        "expert_weights_unchanged": {
            criterion: float(weights[i])
            for i, criterion in enumerate(CRITERIA)
        },

        "expert_consistency_ratio":
            float(
                expert_qa["consistency_ratio"]
            ),

        "scenario_interpretation": (
            "Twelve hypothetical nonexpert "
            "weight-transfer tests plus the "
            "unmodified engineer AHP baseline."
        ),

        "scenario_construction": {
            "description": (
                "Transfer 5 or 10 percentage points "
                "between selected criteria while "
                "preserving total weight = 1."
            ),
            "transfer_points": [5, 10],
            "ordered_transfer_pairs": 6,
            "hypothetical_scenarios": 12,
            "total_including_baseline": 13,
        },

        "grid_count":
            EXPECTED_N,

        "top_decile_size":
            top10_size,

        "top_quintile_size":
            top20_size,

        "rank_tiebreak": (
            "Descending suitability, then "
            "ascending grid_id"
        ),

        "ranking_sets_are_not_suitability_classes":
            True,

        "overall_robustness_diagnostics":
            diagnostics,

        "limitations": [
            (
                "Hypothetical perturbations measure "
                "weight sensitivity, not expert consensus."
            ),
            (
                "Suitability is based only on selected "
                "C1/C2/C4 criteria, not comprehensive "
                "legal or flood-risk suitability."
            ),
            (
                "No final suitability categories or "
                "planning conflict classes are assigned."
            ),
        ],

        "final_suitability_classification_completed":
            False,

        "final_section03_release_completed":
            False,

        "section04_modified":
            False,

        "source_sha256":
            input_hashes,

        "outputs": {

            "scenario_summary_csv":
                OUT_SUMMARY.relative_to(
                    ROOT
                ).as_posix(),

            "grid_stability_csv":
                OUT_GRID.relative_to(
                    ROOT
                ).as_posix(),

            "scenario_summary_sha256":
                sha256(OUT_SUMMARY),

            "grid_stability_sha256":
                sha256(OUT_GRID),
        },
    }

    OUT_QA.write_text(
        json.dumps(
            qa,
            indent=2,
        ),
        encoding="utf-8",
    )

    # ========================================================
    # STEP 12 - TERMINAL SUMMARY
    # ========================================================

    print("\n" + "=" * 78)

    print(
        "PASS_POST_AHP_SENSITIVITY_WITH_"
        "DOCUMENTED_STRESS_TESTS_"
        "FINAL_RELEASE_PENDING"
    )

    print(
        "Grid cells:", EXPECTED_N
    )

    print(
        "Scenarios:", scenario_count
    )

    print(
        "Worst Spearman correlation:",
        f"{diagnostics['worst_scenario_spearman_correlation']:.6f}",
    )

    print(
        "Minimum top 10% retention:",
        f"{diagnostics['smallest_top_decile_retention_fraction']:.6f}",
    )

    print(
        "Original expert weights modified: NO"
    )

    print(
        "Final classification completed: NO"
    )

    print(
        "Section 04 modified: NO"
    )

    print("\nScenario summary:", OUT_SUMMARY)
    print("Grid-level stability:", OUT_GRID)
    print("QA report:", OUT_QA)

    print("=" * 78)


if __name__ == "__main__":
    main()
