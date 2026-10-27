
"""
STEP 31D - FINAL PRIORITY DECISION MATRIX + DIRECTION + NORMALIZATION
====================================================================

Uses exactly the six indicators frozen in Step 31C.
No new indicator selection, CRITIC weighting, TOPSIS ranking, or model retraining.

All six final criteria are already expressed so that a larger value means a
stronger reason to prioritize a grid. Therefore direction conversion is IDENTITY.

Normalization for CRITIC:
    z = (x - min(x)) / (max(x) - min(x))

Expected output:
    2,194 grids x 6 normalized criteria.
"""

from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd


# =============================================================================
# 0. PATHS
# =============================================================================

ROOT = Path(__file__).resolve().parents[1]

PRIORITY_DIR = ROOT / "outputs" / "priority"
REPORT_DIR = ROOT / "outputs" / "reports"

PRIORITY_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

EXPECTED_GRIDS = 2194
EXPECTED_CRITERIA = 6

STEP31C_AUDIT = REPORT_DIR / "31c_priority_indicator_selection_audit.csv"
STEP31C_SELECTION = PRIORITY_DIR / "31c_final_priority_indicator_selection.csv"
STEP31C_RAW = PRIORITY_DIR / "31c_selected_indicator_raw_diagnostic.csv"

RAW_OUT = PRIORITY_DIR / "31d_priority_decision_matrix_raw.csv"
ORIENTED_OUT = PRIORITY_DIR / "31d_priority_decision_matrix_priority_oriented.csv"
NORMALIZED_OUT = PRIORITY_DIR / "31d_priority_decision_matrix_normalized.csv"
META_OUT = PRIORITY_DIR / "31d_priority_criteria_metadata.csv"
PARAM_OUT = PRIORITY_DIR / "31d_normalization_parameters.csv"
SUMMARY_OUT = PRIORITY_DIR / "31d_decision_matrix_summary.csv"

AUDIT_OUT = REPORT_DIR / "31d_priority_decision_matrix_audit.csv"
METHOD_OUT = REPORT_DIR / "31d_priority_decision_matrix_methodology.json"
REPORT_OUT = REPORT_DIR / "31d_priority_decision_matrix_report.txt"
HASH_OUT = REPORT_DIR / "31d_priority_decision_matrix_hashes.csv"


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
        raise RuntimeError(f"{label} is not fully PASS")


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


# =============================================================================
# 2. VERIFY STEP 31C
# =============================================================================

header("STEP 31D - FINAL PRIORITY DECISION MATRIX + DIRECTION + NORMALIZATION")

require_all_pass(STEP31C_AUDIT, "Step 31C audit")
require(STEP31C_SELECTION, "Step 31C final selection")
require(STEP31C_RAW, "Step 31C raw indicator table")

upstream_hash_before = {
    STEP31C_AUDIT.name: sha256_file(STEP31C_AUDIT),
    STEP31C_SELECTION.name: sha256_file(STEP31C_SELECTION),
    STEP31C_RAW.name: sha256_file(STEP31C_RAW),
}

selection = pd.read_csv(STEP31C_SELECTION, low_memory=False)
raw_source = pd.read_csv(STEP31C_RAW, low_memory=False)

require_columns(
    selection,
    [
        "criterion_order",
        "decision_dimension",
        "selected_candidate_id",
        "final_indicator_name",
        "final_field_name",
        "priority_direction",
        "selection_status",
    ],
    "Step 31C final selection",
)

if len(selection) != EXPECTED_CRITERIA:
    raise RuntimeError(
        f"Expected {EXPECTED_CRITERIA} frozen criteria, found {len(selection)}"
    )

if selection["decision_dimension"].nunique() != EXPECTED_CRITERIA:
    raise RuntimeError("Expected six unique decision dimensions")

if selection["selected_candidate_id"].nunique() != EXPECTED_CRITERIA:
    raise RuntimeError("Expected six unique selected candidate IDs")

if not selection["selection_status"].eq("FINAL_SELECTED_STEP31C").all():
    raise RuntimeError("Step 31C selection is not fully frozen")

selection = selection.sort_values("criterion_order").reset_index(drop=True)
fields = selection["final_field_name"].astype(str).tolist()

require_columns(raw_source, ["grid_id", *fields], "Step 31C raw indicator table")

print("PASS: Step 31C fully verified.")
print(f"Frozen criteria: {len(selection)}")


# =============================================================================
# 3. BUILD RAW MATRIX
# =============================================================================

header("3. BUILD RAW 2,194-GRID DECISION MATRIX")

raw = raw_source[["grid_id", *fields]].copy()
raw["grid_id"] = raw["grid_id"].astype(str)

if len(raw) != EXPECTED_GRIDS:
    raise RuntimeError(f"Rows={len(raw)}, expected={EXPECTED_GRIDS}")

if raw["grid_id"].nunique() != EXPECTED_GRIDS:
    raise RuntimeError(
        f"Unique grids={raw['grid_id'].nunique()}, expected={EXPECTED_GRIDS}"
    )

if raw["grid_id"].duplicated().any():
    raise RuntimeError("Duplicate grid_id values found")

for col in fields:
    raw[col] = pd.to_numeric(raw[col], errors="coerce")

    if raw[col].isna().any():
        raise RuntimeError(f"{col}: missing/non-numeric values found")

    if not np.isfinite(raw[col].to_numpy(dtype=float)).all():
        raise RuntimeError(f"{col}: non-finite values found")

    if raw[col].nunique() <= 1:
        raise RuntimeError(f"{col}: no cross-grid variation")

raw.to_csv(RAW_OUT, index=False)

print(f"Raw decision matrix: {len(raw):,} grids x {len(fields)} criteria")
print("PASS: complete, finite, unique-grid matrix.")


# =============================================================================
# 4. DIRECTION HANDLING
# =============================================================================

header("4. PRIORITY-DIRECTION HANDLING")

allowed_directions = {
    "HIGHER=HIGHER_PRIORITY",
    "MORE_POSITIVE=HIGHER_PRIORITY",
    "HIGHER_BENEFIT=HIGHER_PRIORITY",
}

unexpected = sorted(
    set(selection["priority_direction"].astype(str)) - allowed_directions
)

if unexpected:
    raise RuntimeError(f"Unexpected priority directions: {unexpected}")

# Step 31C already defined every selected indicator so that higher = more priority.
# Therefore no sign inversion / reciprocal conversion is needed.
oriented = raw.copy()
oriented.to_csv(ORIENTED_OUT, index=False)

meta_rows = []

for _, row in selection.iterrows():
    meta_rows.append(
        {
            "criterion_order": int(row["criterion_order"]),
            "criterion_code": f"C{int(row['criterion_order'])}",
            "decision_dimension": row["decision_dimension"],
            "selected_candidate_id": row["selected_candidate_id"],
            "final_indicator_name": row["final_indicator_name"],
            "final_field_name": row["final_field_name"],
            "step31c_priority_direction": row["priority_direction"],
            "mcda_orientation": "HIGHER=STRONGER_PRIORITY",
            "direction_transform": "IDENTITY",
            "normalization": "MIN_MAX_0_1",
        }
    )

metadata = pd.DataFrame(meta_rows)
metadata.to_csv(META_OUT, index=False)

print("All six criteria are already priority-oriented.")
print("Direction transform: IDENTITY for C1-C6.")
print("No criterion was sign-flipped.")


# =============================================================================
# 5. NORMALIZATION
# =============================================================================

header("5. MIN-MAX NORMALIZATION TO [0,1]")

normalized = pd.DataFrame({"grid_id": oriented["grid_id"]})
param_rows = []

for _, meta in metadata.iterrows():
    col = meta["final_field_name"]

    x = oriented[col].astype(float)
    xmin = float(x.min())
    xmax = float(x.max())
    xrange = float(xmax - xmin)

    if xrange <= 0 or not np.isfinite(xrange):
        raise RuntimeError(f"{col}: invalid normalization range")

    z = (x - xmin) / xrange
    normalized[col] = z

    rho = float(
        pd.DataFrame({"raw": x, "norm": z})
        .corr(method="spearman")
        .iloc[0, 1]
    )

    # Hard monotonicity audit:
    # min-max normalization is a positive affine transform whenever xrange > 0.
    # Therefore ordering must be non-decreasing after sorting by the raw values.
    # This is more robust than requiring Spearman to equal exactly 1.0 when
    # there are many ties / floating-point representations.
    sort_idx = np.argsort(
        x.to_numpy(dtype=float),
        kind="mergesort",
    )

    sorted_norm = z.to_numpy(dtype=float)[sort_idx]

    rank_order_preserved = bool(
        np.all(
            np.diff(sorted_norm) >= -1e-12
        )
    )

    param_rows.append(
        {
            "criterion_order": int(meta["criterion_order"]),
            "criterion_code": meta["criterion_code"],
            "final_field_name": col,
            "raw_min": xmin,
            "raw_max": xmax,
            "raw_range": xrange,
            "raw_mean": float(x.mean()),
            "raw_std_population": float(x.std(ddof=0)),
            "normalized_min": float(z.min()),
            "normalized_max": float(z.max()),
            "normalized_mean": float(z.mean()),
            "normalized_std_population": float(z.std(ddof=0)),
            "spearman_raw_vs_normalized": rho,
            "rank_order_preserved": rank_order_preserved,
        }
    )

params = pd.DataFrame(param_rows)

normalized.to_csv(NORMALIZED_OUT, index=False)
params.to_csv(PARAM_OUT, index=False)

print(
    params[
        [
            "criterion_code",
            "final_field_name",
            "raw_min",
            "raw_max",
            "normalized_min",
            "normalized_max",
            "spearman_raw_vs_normalized",
        ]
    ].to_string(index=False)
)


# =============================================================================
# 6. SUMMARY
# =============================================================================

header("6. DECISION-MATRIX SUMMARY")

summary_rows = []

for _, meta in metadata.iterrows():
    col = meta["final_field_name"]
    x = raw[col]
    z = normalized[col]

    summary_rows.append(
        {
            "criterion_order": int(meta["criterion_order"]),
            "criterion_code": meta["criterion_code"],
            "decision_dimension": meta["decision_dimension"],
            "selected_candidate_id": meta["selected_candidate_id"],
            "final_field_name": col,
            "raw_min": float(x.min()),
            "raw_max": float(x.max()),
            "raw_mean": float(x.mean()),
            "raw_median": float(x.median()),
            "raw_std_population": float(x.std(ddof=0)),
            "raw_negative_count": int((x < 0).sum()),
            "raw_zero_count": int(np.isclose(x, 0.0, atol=1e-15).sum()),
            "raw_positive_count": int((x > 0).sum()),
            "normalized_min": float(z.min()),
            "normalized_max": float(z.max()),
            "normalized_mean": float(z.mean()),
            "normalized_std_population": float(z.std(ddof=0)),
        }
    )

summary = pd.DataFrame(summary_rows)
summary.to_csv(SUMMARY_OUT, index=False)

print(
    summary[
        [
            "criterion_code",
            "decision_dimension",
            "raw_min",
            "raw_max",
            "raw_negative_count",
            "normalized_min",
            "normalized_max",
        ]
    ].to_string(index=False)
)


# =============================================================================
# 7. STRICT AUDIT
# =============================================================================

header("7. STRICT STEP 31D AUDIT")

audit = []

add_audit(
    audit,
    "step31c_audit",
    True,
    "Step 31C audit verified fully PASS.",
)

add_audit(
    audit,
    "final_criterion_count",
    len(selection) == EXPECTED_CRITERIA,
    f"actual={len(selection)}, expected={EXPECTED_CRITERIA}",
)

add_audit(
    audit,
    "decision_matrix_grid_count",
    len(raw) == EXPECTED_GRIDS
    and raw["grid_id"].nunique() == EXPECTED_GRIDS,
    (
        f"rows={len(raw)}, unique_grids={raw['grid_id'].nunique()}, "
        f"expected={EXPECTED_GRIDS}"
    ),
)

add_audit(
    audit,
    "raw_missing_values",
    raw[fields].isna().sum().sum() == 0,
    "All six raw criteria have zero missing values.",
)

add_audit(
    audit,
    "raw_finite_values",
    bool(np.isfinite(raw[fields].to_numpy(dtype=float)).all()),
    "All raw values are finite.",
)

add_audit(
    audit,
    "all_criteria_have_variation",
    all(raw[c].nunique() > 1 for c in fields),
    "Every criterion varies across grids.",
)

add_audit(
    audit,
    "direction_transform_identity",
    metadata["direction_transform"].eq("IDENTITY").all(),
    "All six were already priority-oriented in Step 31C.",
)

norm_array = normalized[fields].to_numpy(dtype=float)

add_audit(
    audit,
    "normalized_values_finite",
    bool(np.isfinite(norm_array).all()),
    "All normalized values are finite.",
)

add_audit(
    audit,
    "normalized_values_within_0_1",
    bool((norm_array >= -1e-12).all() and (norm_array <= 1 + 1e-12).all()),
    "All normalized values lie inside [0,1].",
)

add_audit(
    audit,
    "each_normalized_min_zero",
    bool(
        np.allclose(
            normalized[fields].min().to_numpy(dtype=float),
            0.0,
            atol=1e-12,
            rtol=0.0,
        )
    ),
    "Each criterion has normalized minimum 0.",
)

add_audit(
    audit,
    "each_normalized_max_one",
    bool(
        np.allclose(
            normalized[fields].max().to_numpy(dtype=float),
            1.0,
            atol=1e-12,
            rtol=0.0,
        )
    ),
    "Each criterion has normalized maximum 1.",
)

add_audit(
    audit,
    "normalization_preserves_rank_order",
    params["rank_order_preserved"].all(),
    (
        "Sorted normalized values are non-decreasing when raw values are "
        "sorted ascending for every criterion. Spearman is retained only "
        "as a diagnostic because ties can create floating-point artifacts."
    ),
)

benefit_row = selection[
    selection["selected_candidate_id"] == "SCEN_COMBINED_G10_MODERATE"
]

if len(benefit_row) != 1:
    raise RuntimeError("Could not uniquely identify final scenario-benefit field")

benefit_field = str(benefit_row.iloc[0]["final_field_name"])

add_audit(
    audit,
    "scenario_benefit_values_preserved_before_normalization",
    bool(
        np.array_equal(
            raw[benefit_field].to_numpy(dtype=float),
            oriented[benefit_field].to_numpy(dtype=float),
        )
    ),
    "No clipping/sign-forcing of modeled cooling benefit.",
)

add_audit(
    audit,
    "no_new_indicator_selection",
    True,
    "Only the six Step31C-frozen criteria are used.",
)

add_audit(
    audit,
    "no_critic_weighting",
    True,
    "CRITIC weighting is deferred to Step31E.",
)

add_audit(
    audit,
    "no_topsis_ranking",
    True,
    "TOPSIS ranking is not performed here.",
)

add_audit(
    audit,
    "no_model_retraining",
    True,
    "No model fitting/retraining performed.",
)

upstream_hash_after = {
    STEP31C_AUDIT.name: sha256_file(STEP31C_AUDIT),
    STEP31C_SELECTION.name: sha256_file(STEP31C_SELECTION),
    STEP31C_RAW.name: sha256_file(STEP31C_RAW),
}

add_audit(
    audit,
    "upstream_step31c_files_unchanged",
    upstream_hash_before == upstream_hash_after,
    "Step31C input hashes are unchanged before vs after Step31D.",
)

audit_df = pd.DataFrame(audit)
audit_df.to_csv(AUDIT_OUT, index=False)

print(audit_df.to_string(index=False))

if not audit_df["status"].eq("PASS").all():
    raise RuntimeError(
        "Step 31D strict audit FAILED. Do not proceed to Step 31E."
    )


# =============================================================================
# 8. METHODOLOGY / REPORT
# =============================================================================

methodology = {
    "step": "31D",
    "title": "Final Priority Decision Matrix + Direction + Normalization",
    "grid_count": EXPECTED_GRIDS,
    "criterion_count": EXPECTED_CRITERIA,
    "source": "six criteria frozen in Step31C",
    "direction_policy": {
        "rule": "higher value = stronger priority for all six criteria",
        "transform": "IDENTITY",
    },
    "normalization": {
        "method": "min-max",
        "formula": "(x-min)/(max-min)",
        "range": "[0,1]",
        "purpose": "CRITIC-ready dimensionless matrix",
    },
    "scenario_benefit_policy": (
        "Negative raw modeled cooling-benefit values are preserved; "
        "no clipping or sign-forcing is applied."
    ),
    "not_done_here": [
        "new indicator selection",
        "CRITIC weighting",
        "TOPSIS ranking",
        "priority classification",
        "model retraining",
    ],
    "next": "Step31E CRITIC weighting",
}

with METHOD_OUT.open("w", encoding="utf-8") as f:
    json.dump(methodology, f, indent=2, ensure_ascii=False)

report_lines = [
    "STEP 31D - FINAL PRIORITY DECISION MATRIX + DIRECTION + NORMALIZATION",
    "=" * 80,
    "",
    "STATUS: COMPLETED",
    f"Grids: {EXPECTED_GRIDS}",
    f"Criteria: {EXPECTED_CRITERIA}",
    "",
    "Direction handling:",
    "- All six criteria already mean: higher value = stronger priority.",
    "- Direction conversion = IDENTITY.",
    "",
    "Normalization:",
    "- Min-max normalization to [0,1].",
    "- Higher normalized value = stronger priority signal.",
    "",
    "Important:",
    "- No new indicator selection.",
    "- Negative modeled intervention benefits are not clipped.",
    "- No CRITIC weighting yet.",
    "- No TOPSIS ranking yet.",
    "- No model retraining.",
    "",
    "NEXT: Step 31E -> CRITIC objective weights.",
]

REPORT_OUT.write_text("\n".join(report_lines), encoding="utf-8")


# =============================================================================
# 9. HASHES
# =============================================================================

hash_targets = [
    RAW_OUT,
    ORIENTED_OUT,
    NORMALIZED_OUT,
    META_OUT,
    PARAM_OUT,
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

header("STEP 31D COMPLETED SUCCESSFULLY")

print(f"Final decision matrix: {EXPECTED_GRIDS:,} grids x {EXPECTED_CRITERIA} criteria")
print("Direction handling: IDENTITY for all six.")
print("Normalization: min-max [0,1].")

print("\nSaved:")
for p in [
    RAW_OUT,
    ORIENTED_OUT,
    NORMALIZED_OUT,
    META_OUT,
    PARAM_OUT,
    SUMMARY_OUT,
    AUDIT_OUT,
    METHOD_OUT,
    REPORT_OUT,
    HASH_OUT,
]:
    print(f"- {p}")

print("\nNo CRITIC weighting performed.")
print("No TOPSIS ranking performed.")
print("No model retraining performed.")
print("NEXT: review Step31D outputs before Step31E.")
