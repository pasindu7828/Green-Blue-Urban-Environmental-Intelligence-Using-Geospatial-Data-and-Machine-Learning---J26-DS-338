"""
STEP 32 - TOPSIS COOLING-PRIORITY RANKING
==========================================

Urban Heat Research - Kaduwela, Sri Lanka

PURPOSE
-------
Rank all 2,194 grid cells using TOPSIS with:

1. the six priority criteria frozen in Step 31C,
2. the priority-oriented min-max normalized matrix verified in Step 31D, and
3. the objective CRITIC weights calculated in Step 31E.

TOPSIS = Technique for Order Preference by Similarity to Ideal Solution.

All six criteria already have the same decision direction:

    higher value = stronger reason for cooling priority

Therefore, for every criterion:

    ideal best  = maximum weighted normalized value
    ideal worst = minimum weighted normalized value

For grid i:

    D_i+ = Euclidean distance to ideal best
    D_i- = Euclidean distance to ideal worst

    TOPSIS_i = D_i- / (D_i+ + D_i-)

Interpretation:

    score closer to 1 -> closer to the high-priority ideal
    score closer to 0 -> closer to the low-priority ideal

IMPORTANT METHODOLOGY DECISIONS
-------------------------------
- The Step-31D min-max normalized matrix is used directly.
- TOPSIS does NOT perform a second normalization here.
- Step-31E CRITIC weights are used exactly as saved.
- No manual/expert weights are mixed in.
- No criterion is added, removed, or re-selected.
- No ML model is fitted or retrained.
- No final High/Medium/Low class thresholds are frozen here.
  Step 32 produces the continuous score and ranking; Step 33 will test
  sensitivity/rank stability before final priority categories are finalized.
- All Step-31D and Step-31E input files are hash-checked before and after.

EXPECTED OUTPUT
---------------
- 2,194 grids
- six frozen criteria
- one TOPSIS score per grid in [0,1]
- complete cooling-priority ranking

NEXT
----
Step 33 -> priority sensitivity / rank-stability validation.
"""

from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd


# =============================================================================
# 0. PATHS / SETTINGS
# =============================================================================

ROOT = Path(__file__).resolve().parents[1]

PRIORITY_DIR = ROOT / "outputs" / "priority"
REPORT_DIR = ROOT / "outputs" / "reports"

PRIORITY_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

EXPECTED_GRIDS = 2194
EXPECTED_CRITERIA = 6
TOL = 1e-12

# Step 31D inputs.
STEP31D_AUDIT = REPORT_DIR / "31d_priority_decision_matrix_audit.csv"
STEP31D_NORMALIZED = PRIORITY_DIR / "31d_priority_decision_matrix_normalized.csv"
STEP31D_METADATA = PRIORITY_DIR / "31d_priority_criteria_metadata.csv"
STEP31D_PARAMS = PRIORITY_DIR / "31d_normalization_parameters.csv"

# Step 31E inputs.
STEP31E_AUDIT = REPORT_DIR / "31e_critic_weighting_audit.csv"
STEP31E_WEIGHTS = PRIORITY_DIR / "31e_critic_weights.csv"
STEP31E_COMPONENTS = PRIORITY_DIR / "31e_critic_information_components.csv"

# Step 32 outputs.
WEIGHTED_MATRIX_OUT = PRIORITY_DIR / "32_topsis_weighted_decision_matrix.csv"
IDEAL_OUT = PRIORITY_DIR / "32_topsis_ideal_solutions.csv"
RANKING_OUT = PRIORITY_DIR / "32_topsis_priority_ranking.csv"
TOP100_OUT = PRIORITY_DIR / "32_topsis_top100_grids.csv"
SUMMARY_OUT = PRIORITY_DIR / "32_topsis_score_summary.csv"

AUDIT_OUT = REPORT_DIR / "32_topsis_priority_ranking_audit.csv"
METHOD_OUT = REPORT_DIR / "32_topsis_priority_ranking_methodology.json"
REPORT_OUT = REPORT_DIR / "32_topsis_priority_ranking_report.txt"
HASH_OUT = REPORT_DIR / "32_topsis_priority_ranking_hashes.csv"


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


# =============================================================================
# 2. VERIFY STEP 31D + STEP 31E
# =============================================================================

header("STEP 32 - TOPSIS COOLING-PRIORITY RANKING")

require_all_pass(STEP31D_AUDIT, "Step 31D audit")
require_all_pass(STEP31E_AUDIT, "Step 31E audit")

for path, label in [
    (STEP31D_NORMALIZED, "Step 31D normalized decision matrix"),
    (STEP31D_METADATA, "Step 31D criterion metadata"),
    (STEP31D_PARAMS, "Step 31D normalization parameters"),
    (STEP31E_WEIGHTS, "Step 31E CRITIC weights"),
    (STEP31E_COMPONENTS, "Step 31E CRITIC information components"),
]:
    require(path, label)

upstream_paths = [
    STEP31D_AUDIT,
    STEP31D_NORMALIZED,
    STEP31D_METADATA,
    STEP31D_PARAMS,
    STEP31E_AUDIT,
    STEP31E_WEIGHTS,
    STEP31E_COMPONENTS,
]

upstream_hash_before = {p.name: sha256_file(p) for p in upstream_paths}

matrix = pd.read_csv(STEP31D_NORMALIZED, low_memory=False)
metadata = pd.read_csv(STEP31D_METADATA, low_memory=False)
params = pd.read_csv(STEP31D_PARAMS, low_memory=False)
weights_df = pd.read_csv(STEP31E_WEIGHTS, low_memory=False)
components = pd.read_csv(STEP31E_COMPONENTS, low_memory=False)

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
    components,
    [
        "criterion_order",
        "criterion_code",
        "final_field_name",
        "critic_weight",
    ],
    "Step 31E information components",
)

metadata = metadata.sort_values("criterion_order").reset_index(drop=True)
weights_df = weights_df.sort_values("criterion_order").reset_index(drop=True)
params = params.sort_values("criterion_order").reset_index(drop=True)
components = components.sort_values("criterion_order").reset_index(drop=True)

if len(metadata) != EXPECTED_CRITERIA:
    raise RuntimeError(
        f"Step 31D metadata has {len(metadata)} criteria; expected {EXPECTED_CRITERIA}."
    )

if len(weights_df) != EXPECTED_CRITERIA:
    raise RuntimeError(
        f"Step 31E weights have {len(weights_df)} criteria; expected {EXPECTED_CRITERIA}."
    )

fields = metadata["final_field_name"].astype(str).tolist()
criterion_codes = metadata["criterion_code"].astype(str).tolist()

require_columns(matrix, ["grid_id", *fields], "Step 31D normalized matrix")

# Exact criterion identity/order agreement across Step 31D and Step 31E.
identity_cols = [
    "criterion_order",
    "criterion_code",
    "decision_dimension",
    "selected_candidate_id",
    "final_indicator_name",
    "final_field_name",
]

meta_identity = metadata[identity_cols].copy().reset_index(drop=True)
weight_identity = weights_df[identity_cols].copy().reset_index(drop=True)

if not meta_identity.equals(weight_identity):
    comparison = pd.concat(
        {
            "step31d": meta_identity,
            "step31e": weight_identity,
        },
        axis=1,
    )
    raise RuntimeError(
        "Step 31D metadata and Step 31E weight definitions do not match exactly.\n"
        f"{comparison.to_string(index=False)}"
    )

print("PASS: Step 31D audit fully PASS.")
print("PASS: Step 31E audit fully PASS.")
print("PASS: exact six-criterion identity/order preserved.")


# =============================================================================
# 3. VALIDATE NORMALIZED MATRIX + WEIGHTS
# =============================================================================

header("3. VALIDATE TOPSIS INPUTS")

matrix = matrix[["grid_id", *fields]].copy()
matrix["grid_id"] = matrix["grid_id"].astype(str)

if len(matrix) != EXPECTED_GRIDS:
    raise RuntimeError(f"Rows={len(matrix)}, expected={EXPECTED_GRIDS}")

if matrix["grid_id"].nunique() != EXPECTED_GRIDS:
    raise RuntimeError(
        f"Unique grids={matrix['grid_id'].nunique()}, expected={EXPECTED_GRIDS}"
    )

if matrix["grid_id"].duplicated().any():
    raise RuntimeError("Duplicate grid_id values found in Step 31D matrix")

for col in fields:
    matrix[col] = pd.to_numeric(matrix[col], errors="coerce")

    if matrix[col].isna().any():
        raise RuntimeError(f"{col}: missing/non-numeric normalized values found")

    x = matrix[col].to_numpy(dtype=float)

    if not np.isfinite(x).all():
        raise RuntimeError(f"{col}: non-finite normalized values found")

    if (x < -TOL).any() or (x > 1.0 + TOL).any():
        raise RuntimeError(f"{col}: normalized values outside [0,1]")

    if np.ptp(x) <= 0:
        raise RuntimeError(f"{col}: no cross-grid variation")

# Verify orientation frozen by Step 31D.
if not metadata["mcda_orientation"].astype(str).eq(
    "HIGHER=STRONGER_PRIORITY"
).all():
    raise RuntimeError(
        "At least one criterion is not HIGHER=STRONGER_PRIORITY. "
        "Do not run benefit-oriented TOPSIS."
    )

if not metadata["direction_transform"].astype(str).eq("IDENTITY").all():
    raise RuntimeError("Unexpected Step 31D direction transform found")

if not metadata["normalization"].astype(str).eq("MIN_MAX_0_1").all():
    raise RuntimeError("Unexpected Step 31D normalization method found")

weights_df["critic_weight"] = pd.to_numeric(
    weights_df["critic_weight"], errors="coerce"
)

if weights_df["critic_weight"].isna().any():
    raise RuntimeError("Missing/non-numeric CRITIC weights found")

weights = weights_df.set_index("final_field_name")["critic_weight"].reindex(fields)

if weights.isna().any():
    raise RuntimeError("Could not align all six CRITIC weights to Step 31D fields")

weight_values = weights.to_numpy(dtype=float)

if not np.isfinite(weight_values).all():
    raise RuntimeError("Non-finite CRITIC weights found")

if (weight_values <= 0).any():
    raise RuntimeError("All six CRITIC weights must be positive")

if not np.isclose(weight_values.sum(), 1.0, atol=TOL, rtol=0.0):
    raise RuntimeError(
        f"CRITIC weight sum={weight_values.sum():.12f}; expected 1.0"
    )

# Cross-check the same weights in the components file.
component_weights = (
    components.set_index("final_field_name")["critic_weight"].reindex(fields)
)

if component_weights.isna().any():
    raise RuntimeError("Could not align CRITIC component weights")

if not np.allclose(
    component_weights.to_numpy(dtype=float),
    weight_values,
    atol=TOL,
    rtol=0.0,
):
    raise RuntimeError("Step 31E weights disagree with information-components file")

print(f"TOPSIS matrix: {len(matrix):,} grids x {len(fields)} criteria")
print(f"CRITIC weight sum: {weight_values.sum():.12f}")
print("All criteria: HIGHER = STRONGER PRIORITY")
print("Normalization reused from Step31D: MIN-MAX [0,1]")
print("No second normalization performed.")


# =============================================================================
# 4. WEIGHTED NORMALIZED DECISION MATRIX
# =============================================================================

header("4. BUILD WEIGHTED NORMALIZED DECISION MATRIX")

weighted_numeric = matrix[fields].mul(weights, axis=1)

if weighted_numeric.isna().any().any():
    raise RuntimeError("Weighted decision matrix contains missing values")

weighted_array = weighted_numeric.to_numpy(dtype=float)

if not np.isfinite(weighted_array).all():
    raise RuntimeError("Weighted decision matrix contains non-finite values")

weighted_out = pd.DataFrame({"grid_id": matrix["grid_id"]})

weighted_column_map = {}

for _, meta in metadata.iterrows():
    code = str(meta["criterion_code"])
    field = str(meta["final_field_name"])
    out_col = f"{code}__weighted__{field}"
    weighted_column_map[field] = out_col
    weighted_out[out_col] = weighted_numeric[field].to_numpy(dtype=float)

weighted_out.to_csv(WEIGHTED_MATRIX_OUT, index=False)

print("Weighted matrix created using Step31E CRITIC weights.")
print(f"Rows: {len(weighted_out):,}")
print(f"Weighted criteria: {len(fields)}")


# =============================================================================
# 5. IDEAL BEST / IDEAL WORST
# =============================================================================

header("5. TOPSIS IDEAL BEST / IDEAL WORST")

# All six criteria are benefit-type PRIORITY criteria after Step 31D orientation.
ideal_best = weighted_numeric.max(axis=0)
ideal_worst = weighted_numeric.min(axis=0)

if not finite_array(ideal_best.values):
    raise RuntimeError("Ideal-best vector contains non-finite values")

if not finite_array(ideal_worst.values):
    raise RuntimeError("Ideal-worst vector contains non-finite values")

if (ideal_best.to_numpy(dtype=float) + TOL < ideal_worst.to_numpy(dtype=float)).any():
    raise RuntimeError("At least one criterion has ideal_best < ideal_worst")

ideal_rows = []

for _, meta in metadata.iterrows():
    field = str(meta["final_field_name"])
    code = str(meta["criterion_code"])
    w = float(weights[field])

    ideal_rows.append(
        {
            "criterion_order": int(meta["criterion_order"]),
            "criterion_code": code,
            "decision_dimension": meta["decision_dimension"],
            "final_indicator_name": meta["final_indicator_name"],
            "final_field_name": field,
            "criterion_type": "BENEFIT_PRIORITY",
            "critic_weight": w,
            "normalized_min": float(matrix[field].min()),
            "normalized_max": float(matrix[field].max()),
            "ideal_worst_weighted": float(ideal_worst[field]),
            "ideal_best_weighted": float(ideal_best[field]),
        }
    )

ideal_df = pd.DataFrame(ideal_rows).sort_values("criterion_order")
ideal_df.to_csv(IDEAL_OUT, index=False)

print(
    ideal_df[
        [
            "criterion_code",
            "decision_dimension",
            "critic_weight",
            "ideal_worst_weighted",
            "ideal_best_weighted",
        ]
    ].to_string(index=False)
)


# =============================================================================
# 6. TOPSIS DISTANCES + CLOSENESS SCORE
# =============================================================================

header("6. TOPSIS DISTANCES + CLOSENESS SCORE")

best_vec = ideal_best.reindex(fields).to_numpy(dtype=float)
worst_vec = ideal_worst.reindex(fields).to_numpy(dtype=float)

# Euclidean distance to positive ideal (high-priority ideal) and negative ideal.
distance_to_best = np.sqrt(
    np.sum((weighted_array - best_vec) ** 2, axis=1)
)

distance_to_worst = np.sqrt(
    np.sum((weighted_array - worst_vec) ** 2, axis=1)
)

denominator = distance_to_best + distance_to_worst

if not finite_array(distance_to_best):
    raise RuntimeError("Distance-to-best contains non-finite values")

if not finite_array(distance_to_worst):
    raise RuntimeError("Distance-to-worst contains non-finite values")

if (distance_to_best < -TOL).any() or (distance_to_worst < -TOL).any():
    raise RuntimeError("TOPSIS distances cannot be negative")

if (denominator <= TOL).any():
    bad_count = int((denominator <= TOL).sum())
    raise RuntimeError(
        f"TOPSIS closeness denominator is zero/near-zero for {bad_count} grids"
    )

score = distance_to_worst / denominator

# Numerical protection only; values should already lie in [0,1].
if (score < -TOL).any() or (score > 1.0 + TOL).any():
    raise RuntimeError("TOPSIS score outside [0,1]")

score = np.clip(score, 0.0, 1.0)

ranking = pd.DataFrame(
    {
        "grid_id": matrix["grid_id"].astype(str),
        "distance_to_ideal_best": distance_to_best,
        "distance_to_ideal_worst": distance_to_worst,
        "topsis_priority_score": score,
    }
)

# Exact score ties are kept visible. A deterministic ordinal rank is also
# supplied for operational sorting; grid_id is used ONLY as a tie-break when
# two grids have exactly equal TOPSIS scores.
ranking["score_tie_size"] = (
    ranking.groupby("topsis_priority_score")["grid_id"].transform("size").astype(int)
)

ranking["topsis_rank_with_ties"] = (
    ranking["topsis_priority_score"]
    .rank(method="min", ascending=False)
    .astype(int)
)

ranking = ranking.sort_values(
    ["topsis_priority_score", "grid_id"],
    ascending=[False, True],
    kind="mergesort",
).reset_index(drop=True)

ranking["topsis_rank"] = np.arange(1, len(ranking) + 1, dtype=int)

if len(ranking) > 1:
    ranking["priority_percentile"] = (
        100.0 * (len(ranking) - ranking["topsis_rank"]) / (len(ranking) - 1)
    )
else:
    ranking["priority_percentile"] = 100.0

# Bring ranking columns into a readable order.
ranking = ranking[
    [
        "grid_id",
        "topsis_priority_score",
        "topsis_rank",
        "topsis_rank_with_ties",
        "priority_percentile",
        "score_tie_size",
        "distance_to_ideal_best",
        "distance_to_ideal_worst",
    ]
]

ranking.to_csv(RANKING_OUT, index=False)
ranking.head(100).to_csv(TOP100_OUT, index=False)

score_values = ranking["topsis_priority_score"].to_numpy(dtype=float)

summary = pd.DataFrame(
    [
        {
            "grid_count": len(ranking),
            "criterion_count": EXPECTED_CRITERIA,
            "score_min": float(np.min(score_values)),
            "score_q01": float(np.quantile(score_values, 0.01)),
            "score_q05": float(np.quantile(score_values, 0.05)),
            "score_q25": float(np.quantile(score_values, 0.25)),
            "score_median": float(np.median(score_values)),
            "score_mean": float(np.mean(score_values)),
            "score_q75": float(np.quantile(score_values, 0.75)),
            "score_q95": float(np.quantile(score_values, 0.95)),
            "score_q99": float(np.quantile(score_values, 0.99)),
            "score_max": float(np.max(score_values)),
            "score_std_population": float(np.std(score_values, ddof=0)),
            "unique_score_count": int(pd.Series(score_values).nunique()),
            "grids_in_score_ties": int((ranking["score_tie_size"] > 1).sum()),
            "max_tie_size": int(ranking["score_tie_size"].max()),
        }
    ]
)
summary.to_csv(SUMMARY_OUT, index=False)

print(
    "TOPSIS score range: "
    f"{float(np.min(score_values)):.6f} to {float(np.max(score_values)):.6f}"
)
print(
    f"Mean={float(np.mean(score_values)):.6f}, "
    f"Median={float(np.median(score_values)):.6f}"
)
print(
    f"Exact-score ties: grids_in_ties={int((ranking['score_tie_size'] > 1).sum())}, "
    f"max_tie_size={int(ranking['score_tie_size'].max())}"
)

print("\nTop 10 cooling-priority grids:")
print(
    ranking.head(10)[
        [
            "topsis_rank",
            "grid_id",
            "topsis_priority_score",
            "priority_percentile",
            "distance_to_ideal_best",
            "distance_to_ideal_worst",
        ]
    ].to_string(index=False)
)


# =============================================================================
# 7. STRICT STEP 32 AUDIT
# =============================================================================

header("7. STRICT STEP 32 AUDIT")

audit = []

add_audit(
    audit,
    "step31d_audit",
    True,
    "Step 31D audit verified fully PASS.",
)

add_audit(
    audit,
    "step31e_audit",
    True,
    "Step 31E audit verified fully PASS.",
)

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
    "criterion_identity_matches_step31d_step31e",
    meta_identity.equals(weight_identity),
    "Step31D metadata and Step31E CRITIC weight definitions match exactly.",
)

norm_array = matrix[fields].to_numpy(dtype=float)

add_audit(
    audit,
    "normalized_matrix_complete",
    bool(not matrix[fields].isna().any().any() and np.isfinite(norm_array).all()),
    "No missing or non-finite normalized criterion values.",
)

add_audit(
    audit,
    "normalized_values_within_0_1",
    bool((norm_array >= -TOL).all() and (norm_array <= 1.0 + TOL).all()),
    "All Step31D normalized values lie inside [0,1].",
)

add_audit(
    audit,
    "all_criteria_priority_oriented",
    metadata["mcda_orientation"].astype(str).eq("HIGHER=STRONGER_PRIORITY").all(),
    "All six criteria use higher value = stronger cooling priority.",
)

add_audit(
    audit,
    "no_second_normalization",
    True,
    "TOPSIS directly reuses the verified Step31D min-max normalized matrix.",
)

add_audit(
    audit,
    "weights_finite_positive",
    bool(np.isfinite(weight_values).all() and (weight_values > 0).all()),
    "All six Step31E CRITIC weights are finite and positive.",
)

add_audit(
    audit,
    "weight_sum_one",
    bool(np.isclose(weight_values.sum(), 1.0, atol=TOL, rtol=0.0)),
    f"weight_sum={weight_values.sum():.12f}",
)

add_audit(
    audit,
    "weighted_matrix_complete",
    bool(np.isfinite(weighted_array).all()),
    "Weighted normalized matrix contains only finite values.",
)

add_audit(
    audit,
    "ideal_vectors_finite",
    bool(finite_array(best_vec) and finite_array(worst_vec)),
    "Ideal-best and ideal-worst vectors are finite.",
)

add_audit(
    audit,
    "ideal_best_not_below_ideal_worst",
    bool(np.all(best_vec + TOL >= worst_vec)),
    "Every benefit-priority criterion has ideal_best >= ideal_worst.",
)

add_audit(
    audit,
    "distances_finite_nonnegative",
    bool(
        np.isfinite(distance_to_best).all()
        and np.isfinite(distance_to_worst).all()
        and (distance_to_best >= -TOL).all()
        and (distance_to_worst >= -TOL).all()
    ),
    "All Euclidean ideal distances are finite and non-negative.",
)

add_audit(
    audit,
    "closeness_denominator_positive",
    bool((denominator > TOL).all()),
    "D_plus + D_minus is positive for every grid.",
)

add_audit(
    audit,
    "topsis_scores_finite_within_0_1",
    bool(
        np.isfinite(score_values).all()
        and (score_values >= -TOL).all()
        and (score_values <= 1.0 + TOL).all()
    ),
    "All TOPSIS priority scores are finite and lie inside [0,1].",
)

add_audit(
    audit,
    "ranking_complete",
    bool(
        len(ranking) == EXPECTED_GRIDS
        and ranking["grid_id"].nunique() == EXPECTED_GRIDS
        and ranking["topsis_rank"].nunique() == EXPECTED_GRIDS
        and set(ranking["topsis_rank"].tolist()) == set(range(1, EXPECTED_GRIDS + 1))
    ),
    "Deterministic operational rank covers exactly 1..2194 with one row per grid.",
)

add_audit(
    audit,
    "ranking_score_monotonic",
    bool(
        np.all(
            np.diff(ranking["topsis_priority_score"].to_numpy(dtype=float))
            <= TOL
        )
    ),
    "TOPSIS scores are non-increasing from rank 1 to rank 2194.",
)

add_audit(
    audit,
    "no_manual_weights_mixed",
    True,
    "Only Step31E CRITIC weights are applied.",
)

add_audit(
    audit,
    "no_new_indicator_selection",
    True,
    "Exactly the six Step31C-frozen indicators are retained.",
)

add_audit(
    audit,
    "no_priority_class_thresholds_frozen",
    True,
    "Step32 creates continuous scores/ranks only; final class thresholds are deferred until sensitivity validation.",
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
    "upstream_step31d_step31e_files_unchanged",
    upstream_hash_before == upstream_hash_after,
    "All Step31D/Step31E input hashes are unchanged before vs after Step32.",
)

audit_df = pd.DataFrame(audit)
audit_df.to_csv(AUDIT_OUT, index=False)

print(audit_df.to_string(index=False))

if not audit_df["status"].eq("PASS").all():
    raise RuntimeError(
        "Step 32 strict audit FAILED. Do not proceed to Step 33."
    )


# =============================================================================
# 8. METHODOLOGY / REPORT
# =============================================================================

methodology = {
    "step": "32",
    "title": "TOPSIS Cooling-Priority Ranking",
    "study_area": "Kaduwela, Sri Lanka",
    "grid_count": EXPECTED_GRIDS,
    "criterion_count": EXPECTED_CRITERIA,
    "inputs": {
        "normalized_matrix": STEP31D_NORMALIZED.name,
        "criterion_metadata": STEP31D_METADATA.name,
        "weights": STEP31E_WEIGHTS.name,
    },
    "normalization": {
        "source": "Step31D",
        "method": "min-max [0,1]",
        "second_normalization_in_step32": False,
        "reason": (
            "Step31D already produced the frozen, dimensionless, priority-oriented "
            "matrix used for CRITIC and TOPSIS."
        ),
    },
    "criterion_orientation": {
        "all_criteria": "benefit-type priority criteria",
        "meaning": "higher value = stronger cooling-priority need",
        "direction_transform": "IDENTITY in Step31D",
    },
    "weights": {
        "method": "CRITIC objective weights",
        "source": "Step31E",
        "manual_weights_mixed": False,
    },
    "topsis": {
        "weighted_matrix": "v_ij = normalized_ij * weight_j",
        "ideal_best": "column-wise maximum of weighted matrix",
        "ideal_worst": "column-wise minimum of weighted matrix",
        "distance": "Euclidean",
        "distance_to_best": "sqrt(sum_j((v_ij - v_j_plus)^2))",
        "distance_to_worst": "sqrt(sum_j((v_ij - v_j_minus)^2))",
        "closeness": "D_minus / (D_plus + D_minus)",
        "ranking": "descending TOPSIS closeness score",
        "score_interpretation": "higher score = stronger cooling priority",
    },
    "tie_policy": {
        "scientific_tie_rank": "minimum rank retained in topsis_rank_with_ties",
        "operational_rank": (
            "deterministic sequential rank after descending score; grid_id is "
            "used only to order exact score ties"
        ),
    },
    "priority_classification": {
        "performed": False,
        "reason": (
            "Step33 will evaluate weight/rank sensitivity before final "
            "High/Medium/Low or similar classes are frozen."
        ),
    },
    "model_retrained": False,
    "next": "Step33 priority sensitivity / rank-stability validation",
}

with METHOD_OUT.open("w", encoding="utf-8") as f:
    json.dump(methodology, f, indent=2, ensure_ascii=False)

report_lines = [
    "STEP 32 - TOPSIS COOLING-PRIORITY RANKING",
    "=" * 64,
    "",
    "STATUS: COMPLETED",
    "",
    f"Grids    : {EXPECTED_GRIDS}",
    f"Criteria : {EXPECTED_CRITERIA}",
    "",
    "INPUTS:",
    "- Step31D min-max normalized, priority-oriented decision matrix",
    "- Step31E objective CRITIC weights",
    "",
    "TOPSIS FORMULA:",
    "v_ij = normalized_ij * CRITIC_weight_j",
    "D_plus  = Euclidean distance to high-priority ideal best",
    "D_minus = Euclidean distance to low-priority ideal worst",
    "score = D_minus / (D_plus + D_minus)",
    "",
    "INTERPRETATION:",
    "- higher score = closer to high-priority ideal",
    "- lower score = closer to low-priority ideal",
    "",
    "CRITIC WEIGHTS USED:",
]

for _, row in weights_df.sort_values("criterion_order").iterrows():
    report_lines.append(
        f"{row['criterion_code']} | {row['decision_dimension']} | "
        f"weight={float(row['critic_weight']):.8f} | "
        f"{float(row['critic_weight_pct']):.4f}%"
    )

report_lines += [
    "",
    "TOPSIS SCORE SUMMARY:",
    f"min={float(np.min(score_values)):.8f}",
    f"mean={float(np.mean(score_values)):.8f}",
    f"median={float(np.median(score_values)):.8f}",
    f"max={float(np.max(score_values)):.8f}",
    f"unique_scores={int(pd.Series(score_values).nunique())}",
    f"grids_in_exact_score_ties={int((ranking['score_tie_size'] > 1).sum())}",
    "",
    "TOP 10 GRIDS:",
]

for _, row in ranking.head(10).iterrows():
    report_lines.append(
        f"rank={int(row['topsis_rank'])} | grid_id={row['grid_id']} | "
        f"score={float(row['topsis_priority_score']):.8f}"
    )

report_lines += [
    "",
    "IMPORTANT:",
    "- no second normalization was performed",
    "- no manual/expert weights were mixed in",
    "- no criterion was added/removed/re-selected",
    "- no final priority class thresholds were frozen",
    "- no predictive model was retrained",
    "",
    "NEXT:",
    "Step33 -> sensitivity and rank-stability validation.",
]

REPORT_OUT.write_text("\n".join(report_lines), encoding="utf-8")


# =============================================================================
# 9. HASHES
# =============================================================================

hash_targets = [
    WEIGHTED_MATRIX_OUT,
    IDEAL_OUT,
    RANKING_OUT,
    TOP100_OUT,
    SUMMARY_OUT,
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
# 10. FINAL
# =============================================================================

header("STEP 32 COMPLETED SUCCESSFULLY")

print(f"TOPSIS ranking completed for {EXPECTED_GRIDS:,} grids.")
print(f"Criteria used: {EXPECTED_CRITERIA}")
print("Weights: Step31E CRITIC objective weights")
print("Higher TOPSIS score = stronger cooling priority")

print("\nTop 10:")
print(
    ranking.head(10)[
        [
            "topsis_rank",
            "grid_id",
            "topsis_priority_score",
            "priority_percentile",
        ]
    ].to_string(index=False)
)

print("\nSaved:")
for p in [
    WEIGHTED_MATRIX_OUT,
    IDEAL_OUT,
    RANKING_OUT,
    TOP100_OUT,
    SUMMARY_OUT,
    AUDIT_OUT,
    METHOD_OUT,
    REPORT_OUT,
    HASH_OUT,
]:
    print(f"- {p}")

print("\nNo priority-class thresholds frozen in Step32.")
print("No model retraining performed.")
print("NEXT: review Step32 outputs before Step33 sensitivity validation.")
