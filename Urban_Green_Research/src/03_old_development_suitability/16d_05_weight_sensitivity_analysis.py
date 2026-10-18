
"""
STEP 16D-E - NON-EXPERT CRITERION-WEIGHT SENSITIVITY

Exploratory 3-criterion weight grid for:
C1 = Road proximity
C2 = Road density (500 m)
C4 = Slope

Tests every weight triple in 0.05 increments that:
- Sums to 1.00
- Assigns each criterion at least 0.10

IMPORTANT:
- NOT expert AHP weights.
- NOT final development suitability.
- NOT official planning policy.
- No C3 or C5 suitability contribution.
- All frozen research inputs remain unchanged.
"""

from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import os

import geopandas as gpd
import numpy as np
import pandas as pd


# ============================================================
# 1. PROJECT PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

REPORTS = ROOT / "outputs/reports"

STANDARD = (
    REPORTS
    / "16d_a_provisional_standardized_criteria_100m.csv"
)

BASE = (
    REPORTS
    / "16d_b_provisional_suitability_baseline_100m.csv"
)

BASE_GPKG = (
    REPORTS
    / "16d_b_provisional_suitability_baseline_100m.gpkg"
)

A_SUMMARY = (
    REPORTS
    / "16d_a_standardized_criteria_summary.json"
)

B_SUMMARY = (
    REPORTS
    / "16d_b_provisional_baseline_summary.json"
)

C_SUMMARY = (
    REPORTS
    / "16d_c_spatial_baseline_summary.json"
)

D_SUMMARY = (
    REPORTS
    / "16d_d_ahp_consistency_summary.json"
)


# ============================================================
# 2. OUTPUT FILES
# ============================================================

OUT_SCENARIOS = (
    REPORTS
    / "16d_e_weight_sensitivity_scenarios.csv"
)

OUT_GRID = (
    REPORTS
    / "16d_e_grid_weight_robustness_100m.csv"
)

OUT_GPKG = (
    REPORTS
    / "16d_e_grid_weight_robustness_100m.gpkg"
)

OUT_METHODS = (
    REPORTS
    / "16d_e_weight_sensitivity_method_summary.csv"
)

OUT_QA = (
    REPORTS
    / "16d_e_weight_sensitivity_qa.csv"
)

OUT_SUMMARY = (
    REPORTS
    / "16d_e_weight_sensitivity_summary.json"
)


# ============================================================
# 3. CONFIGURATION
# ============================================================

N = 9240

CRITERIA = ["C1", "C2", "C4"]

PERCENTILE_COLS = [
    "s_c1_road_pc",
    "s_c2_road_density_pc",
    "s_c4_slope_pc",
]

RANK_COLS = [
    "sens_c1_rank",
    "sens_c2_rank",
    "sens_c4_rank",
]

# Weight values are multiples of 0.05.
SEED_GRID_UNIT = 20

# Minimum weight = 2/20 = 0.10.
MIN_PARTS = 2

# Descriptive robustness-frequency threshold only.
# NOT a prediction probability or legal threshold.
ROBUST_FREQUENCY_CUTOFF = 0.75


# ============================================================
# 4. HELPER FUNCTIONS
# ============================================================

def digest(path):

    h = hashlib.sha256()

    with path.open("rb") as source:

        for chunk in iter(
            lambda: source.read(1024 * 1024),
            b"",
        ):
            h.update(chunk)

    return h.hexdigest()


def load_json(path):

    with path.open(encoding="utf-8") as source:
        return json.load(source)


def read_ids(path):

    df = pd.read_csv(
        path,
        dtype={"grid_id": "string"},
    )

    df["grid_id"] = (
        df["grid_id"].str.strip()
    )

    return df


def as_boolean(series):

    x = (
        series.astype("string")
        .str.strip()
        .str.lower()
    )

    accepted = x.isin([
        "true",
        "false",
        "1",
        "0",
    ])

    if not bool(accepted.all()):

        raise ValueError(
            f"Invalid boolean values in {series.name}"
        )

    return x.isin([
        "true",
        "1",
    ]).to_numpy(dtype=bool)


def top_tenth(values):
    """
    Upper 10% of scores.
    Includes all ties at the cutoff.
    """

    threshold = np.quantile(
        values,
        0.90,
        axis=0,
    )

    return values >= (
        threshold - 1e-12
    )


# ============================================================
# 5. GENERATE EXPLORATORY WEIGHT SCENARIOS
# ============================================================

def weight_grid():

    triples = []

    for a in range(
        MIN_PARTS,
        SEED_GRID_UNIT - 2 * MIN_PARTS + 1,
    ):

        for b in range(
            MIN_PARTS,
            SEED_GRID_UNIT - a - MIN_PARTS + 1,
        ):

            c = SEED_GRID_UNIT - a - b

            if c >= MIN_PARTS:

                triples.append(
                    (a, b, c)
                )

    return (
        np.array(
            triples,
            dtype=float,
        )
        / SEED_GRID_UNIT
    )


# ============================================================
# 6. ANALYZE ONE STANDARDIZATION METHOD
# ============================================================

def analyze_method(
    method,
    X,
    reference,
    top_reference,
    weights,
    review,
    conservation,
):

    # Shape:
    # 9240 grid cells x 120 weight scenarios

    scores = X @ weights.T

    high = top_tenth(
        scores
    )

    ranks = (
        pd.DataFrame(scores)
        .rank(
            method="average",
            axis=0,
        )
        .to_numpy(dtype=float)
    )

    reference_ranks = (
        pd.Series(reference)
        .rank(method="average")
        .to_numpy(dtype=float)
    )

    # --------------------------------------------------------
    # Spearman rank correlation against reference
    # --------------------------------------------------------

    centered_reference = (
        reference_ranks
        - reference_ranks.mean()
    )

    centered_ranks = (
        ranks
        - ranks.mean(
            axis=0,
            keepdims=True,
        )
    )

    denominator = (
        np.sqrt(
            np.sum(centered_reference ** 2)
        )
        * np.sqrt(
            np.sum(
                centered_ranks ** 2,
                axis=0,
            )
        )
    )

    spearman = (
        np.sum(
            centered_reference[:, None]
            * centered_ranks,
            axis=0,
        )
        / denominator
    )

    # --------------------------------------------------------
    # High-ranked grid overlap
    # --------------------------------------------------------

    common_top = (
        high & top_reference[:, None]
    ).sum(axis=0)

    top_union = (
        high | top_reference[:, None]
    ).sum(axis=0)

    top_jaccard = (
        common_top / top_union
    )

    mean_changes = (
        np.mean(
            np.abs(
                scores - reference[:, None]
            ),
            axis=0,
        )
    )

    top_counts = high.sum(
        axis=0
    )

    flagged_counts = (
        high & review[:, None]
    ).sum(axis=0)

    conservation_counts = (
        high & conservation[:, None]
    ).sum(axis=0)

    # --------------------------------------------------------
    # Scenario-level records
    # --------------------------------------------------------

    scenarios = []

    for j, (w1, w2, w4) in enumerate(
        weights
    ):

        scenarios.append({
            "scenario_id":
                f"{method}_W{j + 1:03d}",

            "normalization_method":
                method,

            "w_c1": float(w1),

            "w_c2": float(w2),

            "w_c4": float(w4),

            "weight_source":
                "SYSTEMATIC_EXPLORATORY_GRID_NOT_EXPERT",

            "mean_weighted_score":
                float(scores[:, j].mean()),

            "mean_abs_change_vs_equal_baseline":
                float(mean_changes[j]),

            "spearman_vs_equal_baseline":
                float(spearman[j]),

            "top_decile_count_including_ties":
                int(top_counts[j]),

            "top_decile_jaccard_vs_equal_baseline":
                float(top_jaccard[j]),

            "top_decile_review_flag_count":
                int(flagged_counts[j]),

            "top_decile_review_flag_pct":
                float(
                    100
                    * flagged_counts[j]
                    / top_counts[j]
                ),

            "top_decile_conservation_water_count":
                int(conservation_counts[j]),

            "final_model_status":
                "NOT_FINAL_NOT_AHP",
        })

    # --------------------------------------------------------
    # Per-grid robustness indicators
    # --------------------------------------------------------

    frequency = high.mean(
        axis=1
    )

    cell = {
        f"{method}_top10_frequency":
            frequency,

        f"{method}_top10_in_every_scenario":
            high.all(axis=1),

        f"{method}_top10_in_any_scenario":
            high.any(axis=1),

        f"{method}_top10_frequency_ge_75pct":
            (
                frequency
                >= ROBUST_FREQUENCY_CUTOFF
            ),

        f"{method}_score_min":
            scores.min(axis=1),

        f"{method}_score_max":
            scores.max(axis=1),

        f"{method}_score_mean":
            scores.mean(axis=1),

        f"{method}_score_std":
            scores.std(axis=1),

        f"{method}_rank_range":
            (
                ranks.max(axis=1)
                - ranks.min(axis=1)
            ),
    }

    return (
        pd.DataFrame(scenarios),
        cell,
    )


# ============================================================
# 7. MAIN
# ============================================================

def main():

    print("=" * 93)

    print(
        "STEP 16D-E - CRITERION-WEIGHT "
        "SENSITIVITY (EXPLORATORY, NOT AHP)"
    )

    print("=" * 93)

    sources = [
        STANDARD,
        BASE,
        BASE_GPKG,
        A_SUMMARY,
        B_SUMMARY,
        C_SUMMARY,
        D_SUMMARY,
    ]

    for path in sources:

        if not path.is_file():

            raise FileNotFoundError(
                path
            )

    REPORTS.mkdir(
        parents=True,
        exist_ok=True,
    )

    source_hashes = {
        str(path): digest(path)
        for path in sources
    }

    a = load_json(A_SUMMARY)
    b = load_json(B_SUMMARY)
    c = load_json(C_SUMMARY)
    d = load_json(D_SUMMARY)

    checks = []

    def check(
        description,
        passed,
        detail="",
    ):

        checks.append({
            "check": description,
            "passed": bool(passed),
            "detail": str(detail),
        })

    def halt_if_needed():

        failures = [
            item
            for item in checks
            if not item["passed"]
        ]

        if failures:

            pd.DataFrame(checks).to_csv(
                OUT_QA,
                index=False,
            )

            for item in failures:

                print(
                    "REVIEW:",
                    item["check"],
                    item["detail"],
                )

            raise RuntimeError(
                "Step 16D-E QA requires review. "
                "Inspect the QA CSV."
            )

    # ========================================================
    # 8. VERIFY PREVIOUS STEPS
    # ========================================================

    check(
        "Steps 16D-A/B/C technically passed",

        all(
            item.get("qa_status") == "PASS"
            for item in (a, b, c)
        ),
    )

    check(
        "AHP intake technical QA passed",

        d.get("technical_qa_status")
        == "PASS",
    )

    check(
        "Equal-contribution baseline remains provisional",

        b.get("methodology_status")
        == "PROVISIONAL_REFERENCE_BASELINE_ONLY"

        and b.get(
            "final_suitability_model_calculated"
        ) is False,
    )

    check(
        "Previous baseline contains C1/C2/C4",

        b.get("criterion_set")
        == CRITERIA,
    )

    halt_if_needed()

    # ========================================================
    # 9. LOAD RESEARCH DATA
    # ========================================================

    base = read_ids(
        BASE
    )

    standard = read_ids(
        STANDARD
    )

    geometry = gpd.read_file(
        BASE_GPKG,
        layer="provisional_suitability_baseline",
    )

    geometry["grid_id"] = (
        geometry["grid_id"]
        .astype("string")
        .str.strip()
    )

    check(
        "Three inputs have 9240 unique grid IDs",

        all(
            len(df) == N
            and df["grid_id"].notna().all()
            and df["grid_id"].is_unique

            for df in (
                base,
                standard,
                geometry,
            )
        ),
    )

    check(
        "All three grid ID sets match",

        set(base["grid_id"])
        == set(standard["grid_id"])
        == set(geometry["grid_id"]),
    )

    check(
        "GIS polygon geometries valid and projected",

        geometry.crs is not None

        and geometry.crs.to_epsg() == 32644

        and geometry.geometry.notna().all()

        and (
            ~geometry.geometry.is_empty
        ).all()

        and geometry.geometry.is_valid.all(),
    )

    required_baseline = [
        "baseline_pc",
        "baseline_rank",
        "review_needed",
        "conservation_water_pct",
        "small_boundary_fragment",
        "cell_area_m2",
    ]

    check(
        "Required reference and planning-review columns exist",

        set(required_baseline).issubset(
            base.columns
        ),
    )

    check(
        "All six normalized criterion columns exist",

        set(
            PERCENTILE_COLS + RANK_COLS
        ).issubset(standard.columns),
    )

    halt_if_needed()

    # ========================================================
    # 10. ALIGN DATA BY GRID ID
    # ========================================================

    std = (
        standard
        .set_index("grid_id")
        .reindex(
            base["grid_id"]
        )
    )

    geom = (
        geometry
        .set_index("grid_id")
        .geometry
        .reindex(
            base["grid_id"]
        )
    )

    check(
        "Aligned polygon geometries are complete",

        geom.notna().all(),
    )

    halt_if_needed()

    # ========================================================
    # 11. PREPARE NUMERIC INPUTS
    # ========================================================

    X_pc = (
        std[PERCENTILE_COLS]
        .apply(
            pd.to_numeric,
            errors="coerce",
        )
        .to_numpy(dtype=float)
    )

    X_rank = (
        std[RANK_COLS]
        .apply(
            pd.to_numeric,
            errors="coerce",
        )
        .to_numpy(dtype=float)
    )

    ref_pc = pd.to_numeric(
        base["baseline_pc"],
        errors="coerce",
    ).to_numpy(dtype=float)

    ref_rank = pd.to_numeric(
        base["baseline_rank"],
        errors="coerce",
    ).to_numpy(dtype=float)

    review = as_boolean(
        base["review_needed"]
    )

    conservation_values = pd.to_numeric(
        base["conservation_water_pct"],
        errors="coerce",
    ).to_numpy(dtype=float)

    conservation = (
        conservation_values > 0
    )

    small = as_boolean(
        base["small_boundary_fragment"]
    )

    area = pd.to_numeric(
        base["cell_area_m2"],
        errors="coerce",
    ).to_numpy(dtype=float)

    # ========================================================
    # 12. INITIAL DATA QA
    # ========================================================

    check(
        "All criterion and baseline scores finite and within [0,1]",

        all(
            np.isfinite(values).all()

            and np.all(
                (values >= 0)
                & (values <= 1)
            )

            for values in (
                X_pc,
                X_rank,
                ref_pc,
                ref_rank,
            )
        ),
    )

    check(
        "Reference scores equal C1/C2/C4 means",

        np.allclose(
            X_pc.mean(axis=1),
            ref_pc,
            atol=1e-10,
        )

        and np.allclose(
            X_rank.mean(axis=1),
            ref_rank,
            atol=1e-10,
        ),
    )

    check(
        "Clipped cell areas valid and KMC area preserved",

        np.isfinite(area).all()

        and np.all(area > 0)

        and abs(
            area.sum() / 10000
            - float(a["grid_area_ha"])
        ) <= 0.05,
    )

    check(
        "Planning screening values are finite",

        np.isfinite(
            conservation_values
        ).all(),
    )

    halt_if_needed()

    # ========================================================
    # 13. GENERATE WEIGHT SCENARIOS
    # ========================================================

    weights = weight_grid()

    check(
        "120 unique exploratory weight triples",

        len(weights) == 120

        and len(
            np.unique(
                weights,
                axis=0,
            )
        ) == 120,
    )

    check(
        "Weights sum to one and each is between 0.10 and 0.80",

        np.allclose(
            weights.sum(axis=1),
            1.0,
            atol=1e-12,
        )

        and np.all(
            weights >= 0.10 - 1e-12
        )

        and np.all(
            weights <= 0.80 + 1e-12
        ),
    )

    halt_if_needed()

    # ========================================================
    # 14. ANALYZE BOTH SCORING METHODS
    # ========================================================

    print(
        "\n=== ANALYZING 120 WEIGHT TRIPLES "
        "FOR EACH NORMALIZATION ==="
    )

    scenarios_pc, cells_pc = analyze_method(
        "pctclip",
        X_pc,
        ref_pc,
        top_tenth(ref_pc),
        weights,
        review,
        conservation,
    )

    scenarios_rank, cells_rank = analyze_method(
        "rank",
        X_rank,
        ref_rank,
        top_tenth(ref_rank),
        weights,
        review,
        conservation,
    )

    scenarios = pd.concat(
        [
            scenarios_pc,
            scenarios_rank,
        ],
        ignore_index=True,
    )

    # ========================================================
    # 15. GRID-LEVEL ROBUSTNESS DATASET
    # ========================================================

    grid_out = pd.DataFrame({
        "grid_id":
            base["grid_id"].to_numpy(),

        "baseline_pc_equal_reference":
            ref_pc,

        "baseline_rank_equal_reference":
            ref_rank,

        "planning_review":
            review,

        "conservation_water_review":
            conservation,

        "small_boundary_fragment":
            small,

        "cell_area_m2":
            area,
    })

    for group in (
        cells_pc,
        cells_rank,
    ):

        for key, value in group.items():

            grid_out[key] = value

    # ========================================================
    # 16. VALIDATE WEIGHT ROBUSTNESS
    # ========================================================

    metric_columns = [
        "mean_weighted_score",
        "mean_abs_change_vs_equal_baseline",
        "spearman_vs_equal_baseline",
        "top_decile_jaccard_vs_equal_baseline",
        "top_decile_review_flag_pct",
    ]

    check(
        "Exactly 240 method-weight scenarios produced",

        len(scenarios) == 240

        and scenarios[
            "scenario_id"
        ].is_unique,
    )

    check(
        "All scenario statistics finite",

        np.isfinite(
            scenarios[
                metric_columns
            ].to_numpy(dtype=float)
        ).all(),
    )

    check(
        "Scenario rank correlations and overlaps within valid bounds",

        bool(
            scenarios[
                "spearman_vs_equal_baseline"
            ].between(-1, 1).all()

            and scenarios[
                "top_decile_jaccard_vs_equal_baseline"
            ].between(0, 1).all()

            and scenarios[
                "top_decile_review_flag_pct"
            ].between(0, 100).all()
        ),
    )

    frequency_cols = [
        "pctclip_top10_frequency",
        "rank_top10_frequency",
    ]

    check(
        "Per-cell high-rank frequencies between zero and one",

        bool(
            grid_out[
                frequency_cols
            ].ge(0).all().all()

            and grid_out[
                frequency_cols
            ].le(1).all().all()

            and np.isfinite(
                grid_out[
                    frequency_cols
                ].to_numpy(dtype=float)
            ).all()
        ),
    )

    check(
        "Input sources remain byte-for-byte unchanged",

        all(
            digest(path)
            == source_hashes[str(path)]

            for path in sources
        ),
    )

    halt_if_needed()

    # ========================================================
    # 17. SUMMARY BY NORMALIZATION METHOD
    # ========================================================

    method_records = []

    for name, subset in scenarios.groupby(
        "normalization_method"
    ):

        top_frequency = (
            grid_out[
                f"{name}_top10_frequency"
            ].to_numpy(dtype=float)
        )

        method_records.append({
            "normalization_method":
                name,

            "scenario_count":
                len(subset),

            "minimum_spearman_vs_equal":
                float(
                    subset[
                        "spearman_vs_equal_baseline"
                    ].min()
                ),

            "median_spearman_vs_equal":
                float(
                    subset[
                        "spearman_vs_equal_baseline"
                    ].median()
                ),

            "minimum_top_decile_jaccard":
                float(
                    subset[
                        "top_decile_jaccard_vs_equal_baseline"
                    ].min()
                ),

            "median_top_decile_jaccard":
                float(
                    subset[
                        "top_decile_jaccard_vs_equal_baseline"
                    ].median()
                ),

            "max_mean_score_difference":
                float(
                    subset[
                        "mean_abs_change_vs_equal_baseline"
                    ].max()
                ),

            "top10_in_all_scenarios_cells":
                int(
                    (top_frequency == 1).sum()
                ),

            "top10_in_any_scenario_cells":
                int(
                    (top_frequency > 0).sum()
                ),

            "top10_frequency_at_least_75pct_cells":
                int(
                    (
                        top_frequency >= 0.75
                    ).sum()
                ),

            "scenario_top10_review_pct_min":
                float(
                    subset[
                        "top_decile_review_flag_pct"
                    ].min()
                ),

            "scenario_top10_review_pct_max":
                float(
                    subset[
                        "top_decile_review_flag_pct"
                    ].max()
                ),

            "interpretation":
                "NON_EXPERT_SYSTEMATIC_WEIGHT_PERTURBATIONS_ONLY",
        })

    methods = pd.DataFrame(
        method_records
    )

    check(
        "Both normalization methods summarized",

        set(
            methods["normalization_method"]
        ) == {
            "pctclip",
            "rank",
        },
    )

    check(
        "No C3 or C5 contribution added",

        all(
            column not in grid_out.columns

            for column in [
                "weight_c3",
                "weight_c5",
                "final_suitability",
            ]
        ),
    )

    halt_if_needed()

    # ========================================================
    # 18. PREPARE GIS OUTPUT
    # ========================================================

    mapped = gpd.GeoDataFrame(
        grid_out,

        geometry=gpd.GeoSeries(
            geom.to_numpy(),
            crs=geometry.crs,
        ),

        crs=geometry.crs,
    )

    staging = OUT_GPKG.with_name(
        OUT_GPKG.stem + "_staging.gpkg"
    )

    if staging.exists():
        staging.unlink()

    mapped.to_file(
        staging,
        layer="exploratory_weight_robustness",
        driver="GPKG",
        index=False,
    )

    saved = gpd.read_file(
        staging,
        layer="exploratory_weight_robustness",
    )

    check(
        "Saved GIS has 9240 valid grid geometries",

        len(saved) == N

        and saved["grid_id"].is_unique

        and saved.geometry.is_valid.all(),
    )

    check(
        "Saved GIS retains high-rank frequencies",

        np.allclose(
            saved[
                frequency_cols
            ].to_numpy(dtype=float),

            grid_out[
                frequency_cols
            ].to_numpy(dtype=float),

            atol=1e-10,
        ),
    )

    halt_if_needed()

    del saved

    # ========================================================
    # 19. SAVE OUTPUTS
    # ========================================================

    os.replace(
        staging,
        OUT_GPKG,
    )

    scenarios.to_csv(
        OUT_SCENARIOS,
        index=False,
    )

    grid_out.to_csv(
        OUT_GRID,
        index=False,
    )

    methods.to_csv(
        OUT_METHODS,
        index=False,
    )

    pd.DataFrame(checks).to_csv(
        OUT_QA,
        index=False,
    )

    # ========================================================
    # 20. SUMMARY JSON
    # ========================================================

    summary = {
        "step": "16D-E",

        "qa_status": "PASS",

        "methodology_status":
            "NON_EXPERT_EXPLORATORY_WEIGHT_SENSITIVITY_ONLY",

        "generated_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),

        "grid_cells": N,

        "criteria": CRITERIA,

        "weight_generation": (
            "All 0.05-increment triples summing "
            "to 1 with every weight >= 0.10"
        ),

        "weight_range": [
            0.10,
            0.80,
        ],

        "weight_bound_status":
            "EXPLORATORY_NOT_EXPERT_ELICITED",

        "unique_weight_vectors":
            len(weights),

        "normalization_methods": [
            "pctclip",
            "rank",
        ],

        "total_weight_method_scenarios":
            len(scenarios),

        "reference": (
            "Unchanged equal-contribution "
            "C1/C2/C4 Step 16D-B baseline"
        ),

        "ahp_intake_status_at_run":
            d.get("expert_input_status"),

        "expert_weights_assigned_by_this_script":
            False,

        "final_suitability_score_computed":
            False,

        "uncertainty_definition": (
            "Fraction of tested exploratory weight "
            "vectors ranking a cell in the upper "
            "decile; not model prediction probability"
        ),

        "top_decile_ties_included":
            True,

        "qa_checks":
            len(checks),

        "qa_passed":
            sum(
                item["passed"]
                for item in checks
            ),

        "input_sha256": {
            path.name: source_hashes[str(path)]

            for path in sources
        },

        "limitations": [
            (
                "Weight range and increments are "
                "researcher-defined sensitivity "
                "settings, not expert judgments."
            ),
            (
                "Results only evaluate C1/C2/C4 "
                "and cannot resolve C3 direction "
                "or C5 inclusion."
            ),
            (
                "An upper-decile cell is not "
                "necessarily a legally buildable site."
            ),
            (
                "Planning-review and conservation "
                "indicators are descriptive and "
                "do not alter scores."
            ),
            (
                "Small boundary fragments are "
                "retained and must not be "
                "interpreted as parcels."
            ),
            (
                "External validation and final AHP "
                "or an explicitly justified "
                "alternative remain pending."
            ),
        ],

        "outputs": [
            path.name

            for path in [
                OUT_SCENARIOS,
                OUT_GRID,
                OUT_GPKG,
                OUT_METHODS,
                OUT_QA,
                OUT_SUMMARY,
            ]
        ],
    }

    with OUT_SUMMARY.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            summary,
            file,
            indent=2,
            ensure_ascii=False,
        )

    # ========================================================
    # 21. TERMINAL RESULTS
    # ========================================================

    print(
        "\n=== WEIGHT-SENSITIVITY METHOD SUMMARY ==="
    )

    print(
        methods[
            [
                "normalization_method",
                "scenario_count",
                "minimum_spearman_vs_equal",
                "median_spearman_vs_equal",
                "minimum_top_decile_jaccard",
                "top10_in_all_scenarios_cells",
                "top10_frequency_at_least_75pct_cells",
                "scenario_top10_review_pct_min",
                "scenario_top10_review_pct_max",
            ]
        ]
        .round(4)
        .to_string(index=False)
    )

    print("\n=== QA RESULTS ===")

    for item in checks:

        print(
            (
                "PASS: "
                if item["passed"]
                else "REVIEW: "
            )
            + item["check"]
        )

    passed = sum(
        item["passed"]
        for item in checks
    )

    print(
        f"\nOverall QA: PASS "
        f"({passed}/{len(checks)})"
    )

    print("\n=== OUTPUT REPORTS ===")

    for path in [
        OUT_SCENARIOS,
        OUT_GRID,
        OUT_GPKG,
        OUT_METHODS,
        OUT_QA,
        OUT_SUMMARY,
    ]:

        print(
            path.relative_to(ROOT)
        )

    print(
        "\nWeights are exploratory sensitivity "
        "scenarios, not AHP or final suitability."
    )

    print("=" * 93)


# ============================================================
# 22. EXECUTE
# ============================================================

if __name__ == "__main__":
    main()
