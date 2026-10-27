"""
STEP 33 - COOLING-PRIORITY SENSITIVITY + RANK-STABILITY VALIDATION
==================================================================

Urban Heat Research - Kaduwela, Sri Lanka

PURPOSE
-------
Validate how stable the Step-32 CRITIC-TOPSIS cooling-priority ranking is to
reasonable changes in the Step-31E objective criterion weights.

This step DOES NOT change the frozen six criteria and DOES NOT retrain any
predictive model. It reuses:

1. the six priority criteria frozen in Step 31C,
2. the priority-oriented min-max normalized matrix from Step 31D,
3. the baseline CRITIC weights from Step 31E, and
4. the baseline TOPSIS ranking from Step 32.

SENSITIVITY DESIGN
------------------
A. Controlled one-at-a-time (OAT) perturbations
   - each CRITIC weight is multiplied by 0.80, 0.90, 1.10 and 1.20
   - all six weights are then re-normalized to sum to 1
   - 6 criteria x 4 perturbations = 24 controlled scenarios

B. Simultaneous multi-weight stress test
   - 500 reproducible scenarios
   - each baseline weight is independently multiplied by a factor sampled
     from Uniform(0.80, 1.20)
   - perturbed weights are re-normalized to sum to 1
   - fixed random seed = 42

C. Equal-weight benchmark
   - all six criteria receive weight 1/6
   - this is a benchmark, not a replacement for the CRITIC solution

For every scenario, TOPSIS is recomputed using the SAME Step-31D normalized
matrix and the SAME Step-32 formulation.

RANK-STABILITY METRICS
----------------------
- Spearman rank correlation vs Step-32 baseline
- Kendall tau-b rank correlation vs Step-32 baseline
- Top-10 overlap
- Top-50 overlap
- Top-100 overlap
- Top-10% overlap
- Top-25% overlap
- mean / median / 95th percentile / maximum absolute rank shift
- number of grids shifting by more than 25 / 50 / 100 ranks

IMPORTANT
---------
- Sensitivity scenarios are diagnostic only.
- The original Step-31E CRITIC weights are NOT overwritten.
- The original Step-32 ranking is NOT overwritten.
- Low sensitivity stability, if observed, is a valid research finding and does
  not cause a computational audit failure.
- The strict audit checks calculation integrity and reproducibility, not an
  arbitrary minimum robustness threshold.
- No High/Medium/Low class thresholds are automatically invented in this step.
  Step 34 can export the continuous score/rank/percentile; any display bands
  should be explicitly documented if later required by the dashboard.

NEXT
----
Step 34 -> dashboard-ready data export after Step-33 results are reviewed.
"""

from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd

try:
    from scipy.stats import spearmanr, kendalltau
except ImportError as exc:
    raise ImportError(
        "scipy is required for Step 33. Install it with: pip install scipy"
    ) from exc

try:
    import matplotlib.pyplot as plt
except ImportError as exc:
    raise ImportError(
        "matplotlib is required for Step 33 figures. Install it with: pip install matplotlib"
    ) from exc


# =============================================================================
# 0. PATHS / SETTINGS
# =============================================================================

ROOT = Path(__file__).resolve().parents[1]

PRIORITY_DIR = ROOT / "outputs" / "priority"
REPORT_DIR = ROOT / "outputs" / "reports"
FIGURE_DIR = ROOT / "outputs" / "figures" / "cooling_priority"

PRIORITY_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

EXPECTED_GRIDS = 2194
EXPECTED_CRITERIA = 6
TOL = 1e-12

# Sensitivity configuration.
OAT_MULTIPLIERS = (0.80, 0.90, 1.10, 1.20)
MONTE_CARLO_SCENARIOS = 500
MONTE_CARLO_LOW = 0.80
MONTE_CARLO_HIGH = 1.20
RANDOM_SEED = 42

TOP_K_VALUES = (10, 50, 100)
TOP_FRACTIONS = (0.10, 0.25)
RANK_SHIFT_THRESHOLDS = (25, 50, 100)

# Upstream Step 31D.
STEP31D_AUDIT = REPORT_DIR / "31d_priority_decision_matrix_audit.csv"
STEP31D_NORMALIZED = PRIORITY_DIR / "31d_priority_decision_matrix_normalized.csv"
STEP31D_METADATA = PRIORITY_DIR / "31d_priority_criteria_metadata.csv"
STEP31D_PARAMS = PRIORITY_DIR / "31d_normalization_parameters.csv"

# Upstream Step 31E.
STEP31E_AUDIT = REPORT_DIR / "31e_critic_weighting_audit.csv"
STEP31E_WEIGHTS = PRIORITY_DIR / "31e_critic_weights.csv"

# Upstream Step 32.
STEP32_AUDIT = REPORT_DIR / "32_topsis_priority_ranking_audit.csv"
STEP32_RANKING = PRIORITY_DIR / "32_topsis_priority_ranking.csv"
STEP32_IDEAL = PRIORITY_DIR / "32_topsis_ideal_solutions.csv"

# Step 33 priority outputs.
SCENARIO_WEIGHTS_OUT = PRIORITY_DIR / "33_sensitivity_scenario_weights.csv"
SCENARIO_METRICS_OUT = PRIORITY_DIR / "33_sensitivity_scenario_metrics.csv"
FAMILY_SUMMARY_OUT = PRIORITY_DIR / "33_sensitivity_family_summary.csv"
RANK_MATRIX_OUT = PRIORITY_DIR / "33_sensitivity_rank_matrix.csv"
GRID_STABILITY_OUT = PRIORITY_DIR / "33_grid_rank_stability.csv"
TOP100_STABILITY_OUT = PRIORITY_DIR / "33_baseline_top100_stability.csv"
SUMMARY_OUT = PRIORITY_DIR / "33_priority_sensitivity_summary.csv"

# Step 33 figures.
SPEARMAN_FIG = FIGURE_DIR / "33_monte_carlo_spearman_distribution.png"
TOP100_FIG = FIGURE_DIR / "33_monte_carlo_top100_overlap_distribution.png"
CONTROLLED_FIG = FIGURE_DIR / "33_controlled_sensitivity_metrics.png"

# Step 33 reports.
AUDIT_OUT = REPORT_DIR / "33_priority_sensitivity_validation_audit.csv"
METHOD_OUT = REPORT_DIR / "33_priority_sensitivity_validation_methodology.json"
REPORT_OUT = REPORT_DIR / "33_priority_sensitivity_validation_report.txt"
HASH_OUT = REPORT_DIR / "33_priority_sensitivity_validation_hashes.csv"


# =============================================================================
# 1. HELPERS
# =============================================================================


def header(title):
    print("\n" + "=" * 132)
    print(title)
    print("=" * 132)


def require(path, label):
    if not path.exists():
        raise FileNotFoundError(f"{label} not found:\n{path}")


def require_columns(df, cols, label):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise RuntimeError(f"{label}: missing columns {missing}")


def require_all_pass(path, label):
    require(path, label)
    df = pd.read_csv(path, low_memory=False)

    status_col = next(
        (c for c in df.columns if str(c).strip().lower() == "status"),
        None,
    )
    if status_col is None:
        raise RuntimeError(f"{label}: no status column")

    status = df[status_col].astype(str).str.strip().str.upper()
    if not status.eq("PASS").all():
        failed = df.loc[~status.eq("PASS")]
        raise RuntimeError(
            f"{label} is not fully PASS. Failed rows:\n"
            f"{failed.to_string(index=False)}"
        )


def sha256_file(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def add_audit(rows, check, passed, details):
    rows.append(
        {
            "check": check,
            "status": "PASS" if passed else "FAIL",
            "details": details,
        }
    )


def finite_array(values):
    return bool(np.isfinite(np.asarray(values, dtype=float)).all())


def normalize_weights(values):
    """Return positive finite weights normalized to sum to exactly 1 numerically."""
    arr = np.asarray(values, dtype=float)

    if not np.isfinite(arr).all():
        raise RuntimeError("Sensitivity weight vector contains non-finite values")

    if (arr <= 0).any():
        raise RuntimeError("Sensitivity weight vector must remain strictly positive")

    total = float(arr.sum())
    if total <= TOL:
        raise RuntimeError("Sensitivity weight sum is zero/near-zero")

    arr = arr / total
    return arr


def topsis_from_weights(matrix_array, grid_ids, weights):
    """
    Reproduce the exact Step-32 TOPSIS formulation.

    matrix_array: Step-31D normalized matrix, shape (n_grids, 6)
    weights: positive normalized vector, shape (6,)
    """
    w = normalize_weights(weights)
    weighted = matrix_array * w.reshape(1, -1)

    ideal_best = np.max(weighted, axis=0)
    ideal_worst = np.min(weighted, axis=0)

    d_plus = np.sqrt(np.sum((weighted - ideal_best) ** 2, axis=1))
    d_minus = np.sqrt(np.sum((weighted - ideal_worst) ** 2, axis=1))

    denom = d_plus + d_minus
    if (denom <= TOL).any():
        raise RuntimeError("TOPSIS denominator zero/near-zero in sensitivity scenario")

    score = d_minus / denom

    if not np.isfinite(score).all():
        raise RuntimeError("Non-finite TOPSIS scores in sensitivity scenario")

    if (score < -TOL).any() or (score > 1.0 + TOL).any():
        raise RuntimeError("Sensitivity TOPSIS score outside [0,1]")

    score = np.clip(score, 0.0, 1.0)

    # Match Step 32 deterministic ordering: descending score, grid_id ascending.
    order_df = pd.DataFrame(
        {
            "row_index": np.arange(len(grid_ids), dtype=int),
            "grid_id": np.asarray(grid_ids, dtype=str),
            "score": score,
        }
    ).sort_values(
        ["score", "grid_id"],
        ascending=[False, True],
        kind="mergesort",
    )

    ranks = np.empty(len(grid_ids), dtype=int)
    ranks[order_df["row_index"].to_numpy(dtype=int)] = np.arange(
        1, len(grid_ids) + 1, dtype=int
    )

    return {
        "weights": w,
        "score": score,
        "rank": ranks,
        "distance_to_best": d_plus,
        "distance_to_worst": d_minus,
        "ideal_best": ideal_best,
        "ideal_worst": ideal_worst,
    }


def safe_spearman(a, b):
    result = spearmanr(a, b)
    statistic = getattr(result, "statistic", result[0])
    return float(statistic)


def safe_kendall(a, b):
    result = kendalltau(a, b, variant="b")
    statistic = getattr(result, "statistic", result[0])
    return float(statistic)


def overlap_fraction(base_rank, scenario_rank, k):
    k = int(k)
    base_idx = set(np.flatnonzero(base_rank <= k).tolist())
    scenario_idx = set(np.flatnonzero(scenario_rank <= k).tolist())
    if not base_idx:
        return np.nan
    return float(len(base_idx & scenario_idx) / len(base_idx))


def scenario_metrics(base_rank, scenario_rank):
    abs_shift = np.abs(scenario_rank.astype(int) - base_rank.astype(int))

    metrics = {
        "spearman_rho": safe_spearman(base_rank, scenario_rank),
        "kendall_tau_b": safe_kendall(base_rank, scenario_rank),
        "mean_abs_rank_shift": float(np.mean(abs_shift)),
        "median_abs_rank_shift": float(np.median(abs_shift)),
        "p95_abs_rank_shift": float(np.quantile(abs_shift, 0.95)),
        "max_abs_rank_shift": int(np.max(abs_shift)),
    }

    for k in TOP_K_VALUES:
        metrics[f"top{k}_overlap"] = overlap_fraction(base_rank, scenario_rank, k)

    n = len(base_rank)
    for frac in TOP_FRACTIONS:
        k = int(np.ceil(n * frac))
        label = int(round(frac * 100))
        metrics[f"top{label}pct_k"] = k
        metrics[f"top{label}pct_overlap"] = overlap_fraction(
            base_rank, scenario_rank, k
        )

    for threshold in RANK_SHIFT_THRESHOLDS:
        metrics[f"grids_shift_gt_{threshold}"] = int((abs_shift > threshold).sum())
        metrics[f"pct_grids_shift_gt_{threshold}"] = float(
            100.0 * (abs_shift > threshold).mean()
        )

    return metrics


def save_histogram(values, title, xlabel, output_path, bins=30):
    values = np.asarray(values, dtype=float)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(values, bins=bins)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Scenario count")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# 2. VERIFY UPSTREAM STEPS 31D / 31E / 32
# =============================================================================

header("STEP 33 - COOLING-PRIORITY SENSITIVITY + RANK-STABILITY VALIDATION")

require_all_pass(STEP31D_AUDIT, "Step 31D audit")
require_all_pass(STEP31E_AUDIT, "Step 31E audit")
require_all_pass(STEP32_AUDIT, "Step 32 audit")

for path, label in [
    (STEP31D_NORMALIZED, "Step 31D normalized decision matrix"),
    (STEP31D_METADATA, "Step 31D criterion metadata"),
    (STEP31D_PARAMS, "Step 31D normalization parameters"),
    (STEP31E_WEIGHTS, "Step 31E CRITIC weights"),
    (STEP32_RANKING, "Step 32 TOPSIS ranking"),
    (STEP32_IDEAL, "Step 32 ideal solutions"),
]:
    require(path, label)

upstream_paths = [
    STEP31D_AUDIT,
    STEP31D_NORMALIZED,
    STEP31D_METADATA,
    STEP31D_PARAMS,
    STEP31E_AUDIT,
    STEP31E_WEIGHTS,
    STEP32_AUDIT,
    STEP32_RANKING,
    STEP32_IDEAL,
]

upstream_hash_before = {p.name: sha256_file(p) for p in upstream_paths}

matrix = pd.read_csv(STEP31D_NORMALIZED, low_memory=False)
metadata = pd.read_csv(STEP31D_METADATA, low_memory=False)
params = pd.read_csv(STEP31D_PARAMS, low_memory=False)
weights_df = pd.read_csv(STEP31E_WEIGHTS, low_memory=False)
baseline_ranking = pd.read_csv(STEP32_RANKING, low_memory=False)

require_columns(matrix, ["grid_id"], "Step 31D normalized matrix")
require_columns(
    metadata,
    [
        "criterion_order",
        "criterion_code",
        "decision_dimension",
        "selected_candidate_id",
        "final_indicator_name",
        "final_field_name",
        "mcda_orientation",
        "direction_transform",
        "normalization",
    ],
    "Step 31D metadata",
)
require_columns(
    params,
    [
        "criterion_order",
        "criterion_code",
        "final_field_name",
        "normalized_min",
        "normalized_max",
    ],
    "Step 31D normalization parameters",
)
require_columns(
    weights_df,
    [
        "criterion_order",
        "criterion_code",
        "decision_dimension",
        "selected_candidate_id",
        "final_indicator_name",
        "final_field_name",
        "critic_weight",
        "critic_weight_pct",
        "weight_rank",
    ],
    "Step 31E CRITIC weights",
)
require_columns(
    baseline_ranking,
    [
        "grid_id",
        "topsis_priority_score",
        "topsis_rank",
        "priority_percentile",
    ],
    "Step 32 TOPSIS ranking",
)

metadata = metadata.sort_values("criterion_order").reset_index(drop=True)
params = params.sort_values("criterion_order").reset_index(drop=True)
weights_df = weights_df.sort_values("criterion_order").reset_index(drop=True)

if len(metadata) != EXPECTED_CRITERIA:
    raise RuntimeError(
        f"Step31D metadata has {len(metadata)} criteria; expected {EXPECTED_CRITERIA}"
    )

if len(weights_df) != EXPECTED_CRITERIA:
    raise RuntimeError(
        f"Step31E has {len(weights_df)} weights; expected {EXPECTED_CRITERIA}"
    )

identity_cols = [
    "criterion_order",
    "criterion_code",
    "decision_dimension",
    "selected_candidate_id",
    "final_indicator_name",
    "final_field_name",
]

meta_identity = metadata[identity_cols].reset_index(drop=True)
weight_identity = weights_df[identity_cols].reset_index(drop=True)

if not meta_identity.equals(weight_identity):
    raise RuntimeError("Step31D and Step31E criterion identity/order do not match")

fields = metadata["final_field_name"].astype(str).tolist()
criterion_codes = metadata["criterion_code"].astype(str).tolist()
require_columns(matrix, ["grid_id", *fields], "Step31D normalized matrix")

print("PASS: Step31D audit fully PASS.")
print("PASS: Step31E audit fully PASS.")
print("PASS: Step32 audit fully PASS.")
print("PASS: exact six-criterion identity/order preserved.")


# =============================================================================
# 3. VALIDATE MATRIX / WEIGHTS / BASELINE RANKING
# =============================================================================

header("3. VALIDATE SENSITIVITY INPUTS + REPRODUCE STEP32 BASELINE")

matrix = matrix[["grid_id", *fields]].copy()
matrix["grid_id"] = matrix["grid_id"].astype(str)

if len(matrix) != EXPECTED_GRIDS:
    raise RuntimeError(f"Rows={len(matrix)}, expected={EXPECTED_GRIDS}")

if matrix["grid_id"].nunique() != EXPECTED_GRIDS:
    raise RuntimeError(
        f"Unique grids={matrix['grid_id'].nunique()}, expected={EXPECTED_GRIDS}"
    )

if matrix["grid_id"].duplicated().any():
    raise RuntimeError("Duplicate grid_id values in Step31D matrix")

for col in fields:
    matrix[col] = pd.to_numeric(matrix[col], errors="coerce")
    if matrix[col].isna().any():
        raise RuntimeError(f"{col}: missing/non-numeric normalized values")

matrix_array = matrix[fields].to_numpy(dtype=float)

if not np.isfinite(matrix_array).all():
    raise RuntimeError("Step31D normalized matrix contains non-finite values")

if (matrix_array < -TOL).any() or (matrix_array > 1.0 + TOL).any():
    raise RuntimeError("Step31D normalized matrix contains values outside [0,1]")

if not metadata["mcda_orientation"].astype(str).eq(
    "HIGHER=STRONGER_PRIORITY"
).all():
    raise RuntimeError("All Step31D criteria must be HIGHER=STRONGER_PRIORITY")

if not metadata["direction_transform"].astype(str).eq("IDENTITY").all():
    raise RuntimeError("Unexpected Step31D direction transform")

if not metadata["normalization"].astype(str).eq("MIN_MAX_0_1").all():
    raise RuntimeError("Unexpected Step31D normalization method")

weights_df["critic_weight"] = pd.to_numeric(
    weights_df["critic_weight"], errors="coerce"
)
if weights_df["critic_weight"].isna().any():
    raise RuntimeError("Missing/non-numeric Step31E CRITIC weights")

base_weights = (
    weights_df.set_index("final_field_name")["critic_weight"]
    .reindex(fields)
    .to_numpy(dtype=float)
)
base_weights = normalize_weights(base_weights)

if not np.isclose(base_weights.sum(), 1.0, atol=TOL, rtol=0.0):
    raise RuntimeError("Baseline CRITIC weights do not sum to 1")

# Align Step32 ranking to Step31D row order.
baseline_ranking["grid_id"] = baseline_ranking["grid_id"].astype(str)

if len(baseline_ranking) != EXPECTED_GRIDS:
    raise RuntimeError("Step32 ranking row count mismatch")

if baseline_ranking["grid_id"].nunique() != EXPECTED_GRIDS:
    raise RuntimeError("Step32 ranking unique-grid count mismatch")

baseline_lookup = baseline_ranking.set_index("grid_id")

missing_grids = sorted(set(matrix["grid_id"]) - set(baseline_lookup.index))
extra_grids = sorted(set(baseline_lookup.index) - set(matrix["grid_id"]))
if missing_grids or extra_grids:
    raise RuntimeError(
        f"Step31D/Step32 grid identity mismatch. missing={len(missing_grids)}, "
        f"extra={len(extra_grids)}"
    )

base_rank_saved = (
    baseline_lookup.loc[matrix["grid_id"], "topsis_rank"].to_numpy(dtype=int)
)
base_score_saved = (
    baseline_lookup.loc[matrix["grid_id"], "topsis_priority_score"]
    .to_numpy(dtype=float)
)

base_calc = topsis_from_weights(
    matrix_array=matrix_array,
    grid_ids=matrix["grid_id"].to_numpy(dtype=str),
    weights=base_weights,
)

max_score_diff = float(np.max(np.abs(base_calc["score"] - base_score_saved)))
rank_match = bool(np.array_equal(base_calc["rank"], base_rank_saved))

if max_score_diff > 1e-10:
    raise RuntimeError(
        f"Could not reproduce Step32 TOPSIS scores. max_abs_diff={max_score_diff:.3e}"
    )

if not rank_match:
    raise RuntimeError("Could not reproduce Step32 deterministic ranking exactly")

print(f"Sensitivity matrix: {EXPECTED_GRIDS:,} grids x {EXPECTED_CRITERIA} criteria")
print(f"Baseline CRITIC weight sum: {base_weights.sum():.12f}")
print(f"Step32 baseline reproduction max score diff: {max_score_diff:.3e}")
print("PASS: Step32 baseline ranking reproduced exactly.")


# =============================================================================
# 4. BUILD SENSITIVITY WEIGHT SCENARIOS
# =============================================================================

header("4. BUILD WEIGHT-SENSITIVITY SCENARIOS")

scenario_rows = []
scenario_weight_vectors = []


def append_scenario(
    scenario_id,
    family,
    description,
    vector,
    changed_criterion="",
    multiplier=np.nan,
    random_index=np.nan,
):
    w = normalize_weights(vector)

    row = {
        "scenario_id": scenario_id,
        "scenario_family": family,
        "description": description,
        "changed_criterion": changed_criterion,
        "multiplier": multiplier,
        "random_index": random_index,
        "weight_sum": float(w.sum()),
    }

    for code, field, value in zip(criterion_codes, fields, w):
        row[f"{code}_weight"] = float(value)
        row[f"{code}_field"] = field

    scenario_rows.append(row)
    scenario_weight_vectors.append(w)


# Baseline scenario.
append_scenario(
    "BASE_CRITIC",
    "BASELINE",
    "Frozen Step31E CRITIC weights",
    base_weights,
)

# Controlled one-at-a-time perturbations.
for j, code in enumerate(criterion_codes):
    for mult in OAT_MULTIPLIERS:
        candidate = base_weights.copy()
        candidate[j] *= mult
        pct = int(round((mult - 1.0) * 100))
        sign = "+" if pct > 0 else ""
        append_scenario(
            scenario_id=f"OAT_{code}_{sign}{pct}PCT",
            family="CONTROLLED_OAT",
            description=(
                f"{code} baseline weight multiplied by {mult:.2f}; "
                "all weights renormalized to sum to 1"
            ),
            vector=candidate,
            changed_criterion=code,
            multiplier=float(mult),
        )

# Equal-weight benchmark.
equal_weights = np.full(EXPECTED_CRITERIA, 1.0 / EXPECTED_CRITERIA, dtype=float)
append_scenario(
    "EQUAL_WEIGHT_BENCHMARK",
    "BENCHMARK",
    "Equal weights (1/6 each); diagnostic benchmark only",
    equal_weights,
)

# Reproducible simultaneous perturbation stress test.
rng = np.random.default_rng(RANDOM_SEED)

for i in range(1, MONTE_CARLO_SCENARIOS + 1):
    multipliers = rng.uniform(
        MONTE_CARLO_LOW,
        MONTE_CARLO_HIGH,
        size=EXPECTED_CRITERIA,
    )
    candidate = base_weights * multipliers
    append_scenario(
        scenario_id=f"MC_{i:04d}",
        family="MONTE_CARLO",
        description=(
            f"Independent simultaneous weight multipliers from "
            f"Uniform({MONTE_CARLO_LOW:.2f},{MONTE_CARLO_HIGH:.2f}); "
            "renormalized"
        ),
        vector=candidate,
        random_index=i,
    )

scenario_weights_df = pd.DataFrame(scenario_rows)
scenario_weights_df.to_csv(SCENARIO_WEIGHTS_OUT, index=False)

expected_scenarios = (
    1
    + EXPECTED_CRITERIA * len(OAT_MULTIPLIERS)
    + 1
    + MONTE_CARLO_SCENARIOS
)

if len(scenario_weights_df) != expected_scenarios:
    raise RuntimeError(
        f"Scenario count={len(scenario_weights_df)}, expected={expected_scenarios}"
    )

print(f"Baseline scenarios       : 1")
print(f"Controlled OAT scenarios : {EXPECTED_CRITERIA * len(OAT_MULTIPLIERS)}")
print("Equal-weight benchmark   : 1")
print(f"Monte Carlo scenarios     : {MONTE_CARLO_SCENARIOS}")
print(f"Total scenarios           : {len(scenario_weights_df)}")
print(f"Monte Carlo random seed   : {RANDOM_SEED}")


# =============================================================================
# 5. RECOMPUTE TOPSIS + COMPARE EVERY SCENARIO WITH STEP32
# =============================================================================

header("5. RECOMPUTE TOPSIS + MEASURE RANK STABILITY")

n_scenarios = len(scenario_weights_df)
n_grids = len(matrix)

rank_matrix = np.empty((n_grids, n_scenarios), dtype=np.int32)
metrics_rows = []

for s_idx, (scenario_row, scenario_weights) in enumerate(
    zip(scenario_rows, scenario_weight_vectors)
):
    result = topsis_from_weights(
        matrix_array=matrix_array,
        grid_ids=matrix["grid_id"].to_numpy(dtype=str),
        weights=scenario_weights,
    )

    scenario_rank = result["rank"]
    rank_matrix[:, s_idx] = scenario_rank

    metrics = scenario_metrics(base_rank_saved, scenario_rank)

    metrics_rows.append(
        {
            "scenario_id": scenario_row["scenario_id"],
            "scenario_family": scenario_row["scenario_family"],
            "changed_criterion": scenario_row["changed_criterion"],
            "multiplier": scenario_row["multiplier"],
            "weight_sum": float(result["weights"].sum()),
            "score_min": float(np.min(result["score"])),
            "score_mean": float(np.mean(result["score"])),
            "score_median": float(np.median(result["score"])),
            "score_max": float(np.max(result["score"])),
            **metrics,
        }
    )

    if (s_idx + 1) % 100 == 0 or s_idx == n_scenarios - 1:
        print(f"Processed sensitivity scenarios: {s_idx + 1}/{n_scenarios}")

scenario_metrics_df = pd.DataFrame(metrics_rows)
scenario_metrics_df.to_csv(SCENARIO_METRICS_OUT, index=False)

# Save wide rank matrix for full reproducibility / downstream diagnostics.
rank_matrix_df = pd.DataFrame({"grid_id": matrix["grid_id"].astype(str)})
for idx, scenario_id in enumerate(scenario_weights_df["scenario_id"].astype(str)):
    rank_matrix_df[scenario_id] = rank_matrix[:, idx]
rank_matrix_df.to_csv(RANK_MATRIX_OUT, index=False)

# Baseline must be exactly self-identical.
base_metric = scenario_metrics_df.loc[
    scenario_metrics_df["scenario_id"].eq("BASE_CRITIC")
].iloc[0]

if not np.isclose(float(base_metric["spearman_rho"]), 1.0, atol=TOL, rtol=0.0):
    raise RuntimeError("Baseline self-comparison Spearman correlation is not 1")

if not np.isclose(float(base_metric["kendall_tau_b"]), 1.0, atol=TOL, rtol=0.0):
    raise RuntimeError("Baseline self-comparison Kendall tau-b is not 1")

for k in TOP_K_VALUES:
    if not np.isclose(float(base_metric[f"top{k}_overlap"]), 1.0, atol=TOL, rtol=0.0):
        raise RuntimeError(f"Baseline self-comparison top-{k} overlap is not 1")

print("PASS: baseline sensitivity scenario is self-identical.")


# =============================================================================
# 6. FAMILY-LEVEL SUMMARY
# =============================================================================

header("6. SENSITIVITY SUMMARY BY SCENARIO FAMILY")

metric_cols = [
    "spearman_rho",
    "kendall_tau_b",
    "top10_overlap",
    "top50_overlap",
    "top100_overlap",
    "top10pct_overlap",
    "top25pct_overlap",
    "mean_abs_rank_shift",
    "p95_abs_rank_shift",
    "max_abs_rank_shift",
]

family_rows = []

for family, group in scenario_metrics_df.groupby("scenario_family", sort=False):
    row = {
        "scenario_family": family,
        "scenario_count": len(group),
    }

    for col in metric_cols:
        values = pd.to_numeric(group[col], errors="coerce").to_numpy(dtype=float)
        row[f"{col}__min"] = float(np.nanmin(values))
        row[f"{col}__median"] = float(np.nanmedian(values))
        row[f"{col}__mean"] = float(np.nanmean(values))
        row[f"{col}__max"] = float(np.nanmax(values))

    family_rows.append(row)

family_summary_df = pd.DataFrame(family_rows)
family_summary_df.to_csv(FAMILY_SUMMARY_OUT, index=False)

print(
    family_summary_df[
        [
            "scenario_family",
            "scenario_count",
            "spearman_rho__min",
            "spearman_rho__median",
            "kendall_tau_b__min",
            "top100_overlap__min",
            "top100_overlap__median",
            "mean_abs_rank_shift__median",
            "max_abs_rank_shift__max",
        ]
    ].to_string(index=False)
)


# =============================================================================
# 7. GRID-LEVEL RANK STABILITY
# =============================================================================

header("7. GRID-LEVEL RANK STABILITY")

# Exclude baseline from perturbation-stability summaries.
nonbase_mask = ~scenario_weights_df["scenario_id"].astype(str).eq("BASE_CRITIC").to_numpy()
nonbase_ranks = rank_matrix[:, nonbase_mask]

rank_shift = nonbase_ranks.astype(np.int32) - base_rank_saved.reshape(-1, 1)
abs_rank_shift = np.abs(rank_shift)

stability = pd.DataFrame(
    {
        "grid_id": matrix["grid_id"].astype(str),
        "baseline_rank": base_rank_saved,
        "baseline_topsis_score": base_score_saved,
        "mean_sensitivity_rank": np.mean(nonbase_ranks, axis=1),
        "median_sensitivity_rank": np.median(nonbase_ranks, axis=1),
        "rank_std_population": np.std(nonbase_ranks, axis=1, ddof=0),
        "best_sensitivity_rank": np.min(nonbase_ranks, axis=1),
        "worst_sensitivity_rank": np.max(nonbase_ranks, axis=1),
        "mean_abs_rank_shift": np.mean(abs_rank_shift, axis=1),
        "median_abs_rank_shift": np.median(abs_rank_shift, axis=1),
        "p95_abs_rank_shift": np.quantile(abs_rank_shift, 0.95, axis=1),
        "max_abs_rank_shift": np.max(abs_rank_shift, axis=1),
        "pct_scenarios_in_top10": 100.0 * np.mean(nonbase_ranks <= 10, axis=1),
        "pct_scenarios_in_top50": 100.0 * np.mean(nonbase_ranks <= 50, axis=1),
        "pct_scenarios_in_top100": 100.0 * np.mean(nonbase_ranks <= 100, axis=1),
    }
)

# Useful broader planning cut-offs without inventing categorical labels.
top10pct_k = int(np.ceil(EXPECTED_GRIDS * 0.10))
top25pct_k = int(np.ceil(EXPECTED_GRIDS * 0.25))

stability["pct_scenarios_in_top10pct"] = (
    100.0 * np.mean(nonbase_ranks <= top10pct_k, axis=1)
)
stability["pct_scenarios_in_top25pct"] = (
    100.0 * np.mean(nonbase_ranks <= top25pct_k, axis=1)
)

stability = stability.sort_values("baseline_rank").reset_index(drop=True)
stability.to_csv(GRID_STABILITY_OUT, index=False)

baseline_top100_stability = stability.loc[stability["baseline_rank"] <= 100].copy()
baseline_top100_stability.to_csv(TOP100_STABILITY_OUT, index=False)

print("Grid-level stability table created for all 2,194 grids.")
print("\nBaseline top-10 stability:")
print(
    stability.head(10)[
        [
            "baseline_rank",
            "grid_id",
            "mean_sensitivity_rank",
            "worst_sensitivity_rank",
            "max_abs_rank_shift",
            "pct_scenarios_in_top10",
            "pct_scenarios_in_top50",
            "pct_scenarios_in_top100",
        ]
    ].to_string(index=False)
)


# =============================================================================
# 8. OVERALL SUMMARY + DIAGNOSTIC FIGURES
# =============================================================================

header("8. OVERALL SENSITIVITY SUMMARY")

nonbase_metrics = scenario_metrics_df.loc[
    ~scenario_metrics_df["scenario_id"].eq("BASE_CRITIC")
].copy()
controlled_metrics = scenario_metrics_df.loc[
    scenario_metrics_df["scenario_family"].eq("CONTROLLED_OAT")
].copy()
mc_metrics = scenario_metrics_df.loc[
    scenario_metrics_df["scenario_family"].eq("MONTE_CARLO")
].copy()
equal_metrics = scenario_metrics_df.loc[
    scenario_metrics_df["scenario_id"].eq("EQUAL_WEIGHT_BENCHMARK")
].copy()

summary = pd.DataFrame(
    [
        {
            "grid_count": EXPECTED_GRIDS,
            "criterion_count": EXPECTED_CRITERIA,
            "total_scenarios_including_baseline": n_scenarios,
            "nonbaseline_scenarios": len(nonbase_metrics),
            "controlled_oat_scenarios": len(controlled_metrics),
            "monte_carlo_scenarios": len(mc_metrics),
            "monte_carlo_seed": RANDOM_SEED,
            "monte_carlo_multiplier_low": MONTE_CARLO_LOW,
            "monte_carlo_multiplier_high": MONTE_CARLO_HIGH,
            "all_nonbase_spearman_min": float(nonbase_metrics["spearman_rho"].min()),
            "all_nonbase_spearman_median": float(nonbase_metrics["spearman_rho"].median()),
            "all_nonbase_kendall_min": float(nonbase_metrics["kendall_tau_b"].min()),
            "all_nonbase_kendall_median": float(nonbase_metrics["kendall_tau_b"].median()),
            "all_nonbase_top100_overlap_min": float(nonbase_metrics["top100_overlap"].min()),
            "all_nonbase_top100_overlap_median": float(nonbase_metrics["top100_overlap"].median()),
            "controlled_spearman_min": float(controlled_metrics["spearman_rho"].min()),
            "controlled_kendall_min": float(controlled_metrics["kendall_tau_b"].min()),
            "controlled_top100_overlap_min": float(controlled_metrics["top100_overlap"].min()),
            "mc_spearman_min": float(mc_metrics["spearman_rho"].min()),
            "mc_spearman_median": float(mc_metrics["spearman_rho"].median()),
            "mc_kendall_min": float(mc_metrics["kendall_tau_b"].min()),
            "mc_kendall_median": float(mc_metrics["kendall_tau_b"].median()),
            "mc_top100_overlap_min": float(mc_metrics["top100_overlap"].min()),
            "mc_top100_overlap_median": float(mc_metrics["top100_overlap"].median()),
            "equal_weight_spearman": float(equal_metrics.iloc[0]["spearman_rho"]),
            "equal_weight_kendall": float(equal_metrics.iloc[0]["kendall_tau_b"]),
            "equal_weight_top100_overlap": float(equal_metrics.iloc[0]["top100_overlap"]),
            "baseline_top100_mean_top100_retention_pct": float(
                baseline_top100_stability["pct_scenarios_in_top100"].mean()
            ),
            "baseline_top100_min_top100_retention_pct": float(
                baseline_top100_stability["pct_scenarios_in_top100"].min()
            ),
        }
    ]
)
summary.to_csv(SUMMARY_OUT, index=False)

print(summary.T.to_string(header=False))

# Diagnostic figures focus on the 500 simultaneous perturbation scenarios.
save_histogram(
    mc_metrics["spearman_rho"].to_numpy(dtype=float),
    "Step 33 Monte Carlo rank-correlation stability",
    "Spearman rank correlation vs baseline",
    SPEARMAN_FIG,
    bins=30,
)

save_histogram(
    100.0 * mc_metrics["top100_overlap"].to_numpy(dtype=float),
    "Step 33 Monte Carlo top-100 stability",
    "Top-100 overlap with baseline (%)",
    TOP100_FIG,
    bins=20,
)

# Controlled OAT scenarios: plot Spearman and top-100 overlap together by scenario.
controlled_plot = controlled_metrics.copy()
controlled_plot["top100_overlap_pct"] = 100.0 * controlled_plot["top100_overlap"]

fig, ax = plt.subplots(figsize=(12, 6))
x = np.arange(len(controlled_plot))
ax.plot(x, controlled_plot["spearman_rho"].to_numpy(dtype=float), marker="o", label="Spearman rho")
ax.plot(x, controlled_plot["top100_overlap"].to_numpy(dtype=float), marker="o", label="Top-100 overlap")
ax.set_xticks(x)
ax.set_xticklabels(controlled_plot["scenario_id"].astype(str), rotation=90)
ax.set_ylim(0.0, 1.02)
ax.set_ylabel("Stability metric (0-1)")
ax.set_title("Controlled one-at-a-time weight sensitivity")
ax.grid(axis="y", alpha=0.25)
ax.legend()
fig.tight_layout()
fig.savefig(CONTROLLED_FIG, dpi=200, bbox_inches="tight")
plt.close(fig)


# =============================================================================
# 9. STRICT STEP 33 AUDIT
# =============================================================================

header("9. STRICT STEP 33 AUDIT")

audit = []

add_audit(audit, "step31d_audit", True, "Step31D audit verified fully PASS.")
add_audit(audit, "step31e_audit", True, "Step31E audit verified fully PASS.")
add_audit(audit, "step32_audit", True, "Step32 audit verified fully PASS.")

add_audit(
    audit,
    "grid_count",
    len(matrix) == EXPECTED_GRIDS and matrix["grid_id"].nunique() == EXPECTED_GRIDS,
    f"rows={len(matrix)}, unique_grids={matrix['grid_id'].nunique()}, expected={EXPECTED_GRIDS}",
)

add_audit(
    audit,
    "criterion_count",
    len(fields) == EXPECTED_CRITERIA,
    f"actual={len(fields)}, expected={EXPECTED_CRITERIA}",
)

add_audit(
    audit,
    "criterion_identity_preserved",
    meta_identity.equals(weight_identity),
    "Exact Step31D/Step31E six-criterion identity and order preserved.",
)

add_audit(
    audit,
    "normalized_matrix_complete",
    bool(np.isfinite(matrix_array).all()),
    "Step31D normalized matrix contains only finite values.",
)

add_audit(
    audit,
    "normalized_values_within_0_1",
    bool((matrix_array >= -TOL).all() and (matrix_array <= 1.0 + TOL).all()),
    "All Step31D normalized values lie inside [0,1].",
)

add_audit(
    audit,
    "baseline_weights_positive_sum_one",
    bool((base_weights > 0).all() and np.isclose(base_weights.sum(), 1.0, atol=TOL, rtol=0.0)),
    f"Baseline weight sum={base_weights.sum():.12f}",
)

scenario_weight_cols = [f"{code}_weight" for code in criterion_codes]
scenario_weight_array = scenario_weights_df[scenario_weight_cols].to_numpy(dtype=float)

add_audit(
    audit,
    "all_sensitivity_weights_finite_positive",
    bool(np.isfinite(scenario_weight_array).all() and (scenario_weight_array > 0).all()),
    "Every sensitivity scenario has six finite positive weights.",
)

add_audit(
    audit,
    "all_sensitivity_weight_sums_one",
    bool(
        np.allclose(
            scenario_weights_df["weight_sum"].to_numpy(dtype=float),
            1.0,
            atol=TOL,
            rtol=0.0,
        )
    ),
    "Every sensitivity weight vector is renormalized to sum to 1.",
)

add_audit(
    audit,
    "expected_scenario_count",
    len(scenario_weights_df) == expected_scenarios,
    f"actual={len(scenario_weights_df)}, expected={expected_scenarios}",
)

add_audit(
    audit,
    "baseline_step32_reproduced",
    bool(max_score_diff <= 1e-10 and rank_match),
    f"max_score_diff={max_score_diff:.3e}, exact_rank_match={rank_match}",
)

metrics_numeric_cols = [
    "spearman_rho",
    "kendall_tau_b",
    "top10_overlap",
    "top50_overlap",
    "top100_overlap",
    "top10pct_overlap",
    "top25pct_overlap",
    "mean_abs_rank_shift",
    "median_abs_rank_shift",
    "p95_abs_rank_shift",
    "max_abs_rank_shift",
]

metrics_numeric = scenario_metrics_df[metrics_numeric_cols].to_numpy(dtype=float)

add_audit(
    audit,
    "sensitivity_metrics_finite",
    bool(np.isfinite(metrics_numeric).all()),
    "All primary sensitivity metrics are finite.",
)

correlation_overlap_cols = [
    "spearman_rho",
    "kendall_tau_b",
    "top10_overlap",
    "top50_overlap",
    "top100_overlap",
    "top10pct_overlap",
    "top25pct_overlap",
]
correlation_overlap = scenario_metrics_df[correlation_overlap_cols].to_numpy(dtype=float)

add_audit(
    audit,
    "correlation_overlap_metrics_within_bounds",
    bool((correlation_overlap >= -1.0 - TOL).all() and (correlation_overlap <= 1.0 + TOL).all()),
    "Correlation/overlap diagnostics lie inside mathematically valid bounds.",
)

add_audit(
    audit,
    "rank_matrix_complete",
    bool(
        rank_matrix.shape == (EXPECTED_GRIDS, expected_scenarios)
        and np.isfinite(rank_matrix).all()
    ),
    f"rank_matrix_shape={rank_matrix.shape}",
)

# Every scenario must contain a complete permutation of ranks 1..2194.
expected_rank_set = np.arange(1, EXPECTED_GRIDS + 1, dtype=int)
rank_permutations_valid = True
for col_idx in range(rank_matrix.shape[1]):
    if not np.array_equal(np.sort(rank_matrix[:, col_idx]), expected_rank_set):
        rank_permutations_valid = False
        break

add_audit(
    audit,
    "all_scenario_rankings_complete",
    rank_permutations_valid,
    "Every sensitivity scenario contains exactly one rank 1..2194 per grid.",
)

add_audit(
    audit,
    "grid_stability_complete",
    bool(
        len(stability) == EXPECTED_GRIDS
        and stability["grid_id"].nunique() == EXPECTED_GRIDS
        and not stability.isna().any().any()
    ),
    "Grid-level stability summary is complete for all 2,194 grids.",
)

add_audit(
    audit,
    "baseline_weights_not_overwritten",
    True,
    "Step31E CRITIC weight file is read-only in Step33.",
)

add_audit(
    audit,
    "baseline_ranking_not_overwritten",
    True,
    "Step32 TOPSIS ranking is read-only in Step33.",
)

add_audit(
    audit,
    "no_new_indicator_selection",
    True,
    "Exactly the six Step31C-frozen indicators are retained in every scenario.",
)

add_audit(
    audit,
    "no_priority_class_thresholds_invented",
    True,
    "Step33 validates continuous ranks; no arbitrary High/Medium/Low thresholds are frozen.",
)

add_audit(
    audit,
    "no_model_retraining",
    True,
    "No predictive model fitting or retraining performed.",
)

upstream_hash_after = {p.name: sha256_file(p) for p in upstream_paths}

add_audit(
    audit,
    "upstream_files_unchanged",
    upstream_hash_before == upstream_hash_after,
    "All Step31D/31E/32 input hashes are unchanged before vs after Step33.",
)

audit_df = pd.DataFrame(audit)
audit_df.to_csv(AUDIT_OUT, index=False)

print(audit_df.to_string(index=False))

if not audit_df["status"].eq("PASS").all():
    raise RuntimeError("Step33 strict audit FAILED. Do not proceed to Step34.")


# =============================================================================
# 10. METHODOLOGY + REPORT
# =============================================================================

methodology = {
    "step": "33",
    "title": "Cooling-Priority Sensitivity and Rank-Stability Validation",
    "study_area": "Kaduwela, Sri Lanka",
    "grid_count": EXPECTED_GRIDS,
    "criterion_count": EXPECTED_CRITERIA,
    "baseline": {
        "normalized_matrix": STEP31D_NORMALIZED.name,
        "weights": STEP31E_WEIGHTS.name,
        "ranking": STEP32_RANKING.name,
        "topsis_reproduced_before_sensitivity": True,
        "max_score_difference_vs_step32": max_score_diff,
    },
    "sensitivity_design": {
        "controlled_oat": {
            "multipliers": list(OAT_MULTIPLIERS),
            "scenario_count": EXPECTED_CRITERIA * len(OAT_MULTIPLIERS),
            "procedure": (
                "multiply one CRITIC weight at a time, then renormalize all six "
                "weights to sum to 1"
            ),
        },
        "monte_carlo": {
            "scenario_count": MONTE_CARLO_SCENARIOS,
            "random_seed": RANDOM_SEED,
            "multiplier_distribution": (
                f"independent Uniform({MONTE_CARLO_LOW},{MONTE_CARLO_HIGH})"
            ),
            "procedure": (
                "multiply all six baseline weights simultaneously by independent "
                "bounded factors, then renormalize to sum to 1"
            ),
            "interpretation": "stress-test scenarios, not statistical confidence intervals",
        },
        "equal_weight_benchmark": {
            "performed": True,
            "weight_each": 1.0 / EXPECTED_CRITERIA,
            "interpretation": "alternative benchmark only; does not replace CRITIC",
        },
    },
    "topsis": {
        "normalization_source": "Step31D min-max [0,1]",
        "second_normalization": False,
        "all_criteria_orientation": "higher = stronger cooling priority",
        "distance": "Euclidean",
        "score": "D_minus / (D_plus + D_minus)",
        "ranking": "descending score; grid_id only breaks exact score ties",
    },
    "stability_metrics": [
        "Spearman rho",
        "Kendall tau-b",
        "Top-10 overlap",
        "Top-50 overlap",
        "Top-100 overlap",
        "Top-10% overlap",
        "Top-25% overlap",
        "absolute rank-shift diagnostics",
        "grid-level scenario retention diagnostics",
    ],
    "robustness_threshold_policy": (
        "No arbitrary numerical robustness threshold is used as a computational "
        "PASS/FAIL gate. Step33 reports the observed stability evidence."
    ),
    "priority_classification": {
        "performed": False,
        "reason": (
            "Sensitivity validation does not itself provide a scientifically unique "
            "High/Medium/Low threshold. Continuous score, rank and percentile remain "
            "the primary decision outputs."
        ),
    },
    "model_retrained": False,
    "next": "Step34 dashboard-ready data export after reviewing Step33 sensitivity evidence",
}

with METHOD_OUT.open("w", encoding="utf-8") as f:
    json.dump(methodology, f, indent=2, ensure_ascii=False)

# Identify worst observed scenarios for transparent reporting.
worst_spearman_row = nonbase_metrics.loc[nonbase_metrics["spearman_rho"].idxmin()]
worst_top100_row = nonbase_metrics.loc[nonbase_metrics["top100_overlap"].idxmin()]
worst_mean_shift_row = nonbase_metrics.loc[
    nonbase_metrics["mean_abs_rank_shift"].idxmax()
]

report_lines = [
    "STEP 33 - COOLING-PRIORITY SENSITIVITY + RANK-STABILITY VALIDATION",
    "=" * 76,
    "",
    "STATUS: COMPLETED",
    "",
    f"Grids: {EXPECTED_GRIDS}",
    f"Criteria: {EXPECTED_CRITERIA}",
    f"Total scenarios including baseline: {n_scenarios}",
    f"Controlled OAT scenarios: {len(controlled_metrics)}",
    f"Monte Carlo scenarios: {len(mc_metrics)}",
    "Equal-weight benchmark: 1",
    "",
    "BASELINE VERIFICATION:",
    f"- Step32 maximum score reproduction difference: {max_score_diff:.3e}",
    f"- Step32 exact deterministic rank reproduction: {rank_match}",
    "",
    "CONTROLLED OAT DESIGN:",
    f"- multipliers: {list(OAT_MULTIPLIERS)}",
    "- one criterion perturbed at a time",
    "- all weights renormalized after perturbation",
    "",
    "MONTE CARLO STRESS TEST:",
    f"- scenarios: {MONTE_CARLO_SCENARIOS}",
    f"- fixed seed: {RANDOM_SEED}",
    f"- multiplier range: [{MONTE_CARLO_LOW}, {MONTE_CARLO_HIGH}]",
    "- simultaneous bounded perturbation of all six weights",
    "- diagnostic stress test, not a statistical confidence interval",
    "",
    "OBSERVED STABILITY SUMMARY:",
    f"- controlled min Spearman rho: {float(controlled_metrics['spearman_rho'].min()):.6f}",
    f"- controlled min Kendall tau-b: {float(controlled_metrics['kendall_tau_b'].min()):.6f}",
    f"- controlled min Top-100 overlap: {100.0 * float(controlled_metrics['top100_overlap'].min()):.2f}%",
    f"- Monte Carlo min Spearman rho: {float(mc_metrics['spearman_rho'].min()):.6f}",
    f"- Monte Carlo median Spearman rho: {float(mc_metrics['spearman_rho'].median()):.6f}",
    f"- Monte Carlo min Kendall tau-b: {float(mc_metrics['kendall_tau_b'].min()):.6f}",
    f"- Monte Carlo min Top-100 overlap: {100.0 * float(mc_metrics['top100_overlap'].min()):.2f}%",
    f"- Monte Carlo median Top-100 overlap: {100.0 * float(mc_metrics['top100_overlap'].median()):.2f}%",
    f"- Equal-weight Spearman rho: {float(equal_metrics.iloc[0]['spearman_rho']):.6f}",
    f"- Equal-weight Top-100 overlap: {100.0 * float(equal_metrics.iloc[0]['top100_overlap']):.2f}%",
    "",
    "WORST OBSERVED NON-BASELINE SCENARIOS:",
    (
        f"- lowest Spearman: {worst_spearman_row['scenario_id']} | "
        f"rho={float(worst_spearman_row['spearman_rho']):.6f}"
    ),
    (
        f"- lowest Top-100 overlap: {worst_top100_row['scenario_id']} | "
        f"overlap={100.0 * float(worst_top100_row['top100_overlap']):.2f}%"
    ),
    (
        f"- largest mean absolute rank shift: {worst_mean_shift_row['scenario_id']} | "
        f"mean_shift={float(worst_mean_shift_row['mean_abs_rank_shift']):.3f}"
    ),
    "",
    "BASELINE TOP-100 RETENTION:",
    (
        "- mean percentage of sensitivity scenarios in which a baseline Top-100 "
        f"grid remains Top-100: {float(baseline_top100_stability['pct_scenarios_in_top100'].mean()):.2f}%"
    ),
    (
        "- minimum percentage for any baseline Top-100 grid: "
        f"{float(baseline_top100_stability['pct_scenarios_in_top100'].min()):.2f}%"
    ),
    "",
    "IMPORTANT:",
    "- original Step31E CRITIC weights were not changed",
    "- original Step32 TOPSIS ranking was not changed",
    "- no criterion was added, removed, or re-selected",
    "- no predictive model was retrained",
    "- no arbitrary priority-class thresholds were frozen",
    "- low robustness, if present, is reported rather than hidden",
    "",
    "NEXT:",
    "Review Step33 evidence, then Step34 dashboard-ready export.",
]

REPORT_OUT.write_text("\n".join(report_lines), encoding="utf-8")


# =============================================================================
# 11. HASHES
# =============================================================================

hash_targets = [
    SCENARIO_WEIGHTS_OUT,
    SCENARIO_METRICS_OUT,
    FAMILY_SUMMARY_OUT,
    RANK_MATRIX_OUT,
    GRID_STABILITY_OUT,
    TOP100_STABILITY_OUT,
    SUMMARY_OUT,
    SPEARMAN_FIG,
    TOP100_FIG,
    CONTROLLED_FIG,
    AUDIT_OUT,
    METHOD_OUT,
    REPORT_OUT,
]

hash_df = pd.DataFrame(
    [
        {
            "file": p.name,
            "sha256": sha256_file(p),
        }
        for p in hash_targets
    ]
)
hash_df.to_csv(HASH_OUT, index=False)


# =============================================================================
# 12. FINAL
# =============================================================================

header("STEP 33 COMPLETED SUCCESSFULLY")

print(f"Sensitivity validation completed for {EXPECTED_GRIDS:,} grids.")
print(f"Frozen criteria retained: {EXPECTED_CRITERIA}")
print(f"Sensitivity scenarios including baseline: {n_scenarios}")
print(f"Controlled OAT scenarios: {len(controlled_metrics)}")
print(f"Monte Carlo stress scenarios: {len(mc_metrics)}")
print("Equal-weight benchmark: 1")

print("\nKey diagnostics:")
print(
    f"Controlled min Spearman rho : {float(controlled_metrics['spearman_rho'].min()):.6f}"
)
print(
    f"Controlled min Top100 overlap: {100.0 * float(controlled_metrics['top100_overlap'].min()):.2f}%"
)
print(
    f"Monte Carlo min Spearman rho : {float(mc_metrics['spearman_rho'].min()):.6f}"
)
print(
    f"Monte Carlo median Spearman  : {float(mc_metrics['spearman_rho'].median()):.6f}"
)
print(
    f"Monte Carlo min Top100 overlap: {100.0 * float(mc_metrics['top100_overlap'].min()):.2f}%"
)
print(
    f"Monte Carlo median Top100 overlap: {100.0 * float(mc_metrics['top100_overlap'].median()):.2f}%"
)
print(
    f"Equal-weight Spearman rho    : {float(equal_metrics.iloc[0]['spearman_rho']):.6f}"
)
print(
    f"Equal-weight Top100 overlap  : {100.0 * float(equal_metrics.iloc[0]['top100_overlap']):.2f}%"
)

print("\nSaved:")
for p in [
    SCENARIO_WEIGHTS_OUT,
    SCENARIO_METRICS_OUT,
    FAMILY_SUMMARY_OUT,
    RANK_MATRIX_OUT,
    GRID_STABILITY_OUT,
    TOP100_STABILITY_OUT,
    SUMMARY_OUT,
    SPEARMAN_FIG,
    TOP100_FIG,
    CONTROLLED_FIG,
    AUDIT_OUT,
    METHOD_OUT,
    REPORT_OUT,
    HASH_OUT,
]:
    print(f"- {p}")

print("\nNo predictive model retraining performed.")
print("No frozen criterion changed.")
print("No arbitrary High/Medium/Low thresholds invented.")
print("NEXT: review Step33 results before Step34 dashboard data export.")
