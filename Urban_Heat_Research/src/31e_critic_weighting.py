
"""
STEP 31E - CRITIC OBJECTIVE WEIGHTING
=====================================

Urban Heat Research - Kaduwela, Sri Lanka

PURPOSE
-------
Calculate objective CRITIC weights for the six priority criteria frozen in
Step 31C and normalized in Step 31D.

CRITIC = CRiteria Importance Through Intercriteria Correlation.

For each criterion j:

    information_j
        = std_j * sum_k (1 - corr_jk)

    weight_j
        = information_j / sum_j(information_j)

Interpretation:
- std_j captures contrast / variability across the 2,194 grids.
- (1 - corr_jk) captures conflict / non-redundancy with other criteria.
- A criterion receives a larger CRITIC weight when it both varies meaningfully
  across grids and contributes information not already duplicated by others.

IMPORTANT
---------
- Pearson correlation is used because this is the standard CRITIC formulation.
- Population standard deviation (ddof=0) is used consistently across all six
  normalized criteria.
- The normalized Step-31D matrix is used directly.
- No manual/expert weight is mixed into CRITIC.
- No criterion is removed or re-selected here.
- No TOPSIS ranking is performed here.
- No model is retrained.
- Step-31D files are verified unchanged before/after the calculation.

NEXT
----
Step 32 -> TOPSIS grid priority ranking using the frozen CRITIC weights.
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

STEP31D_AUDIT = (
    REPORT_DIR
    / "31d_priority_decision_matrix_audit.csv"
)

STEP31D_NORMALIZED = (
    PRIORITY_DIR
    / "31d_priority_decision_matrix_normalized.csv"
)

STEP31D_METADATA = (
    PRIORITY_DIR
    / "31d_priority_criteria_metadata.csv"
)

STEP31D_PARAMS = (
    PRIORITY_DIR
    / "31d_normalization_parameters.csv"
)

# Step 31E outputs.
CORR_PATH = (
    PRIORITY_DIR
    / "31e_critic_pearson_correlation_matrix.csv"
)

COMPONENT_PATH = (
    PRIORITY_DIR
    / "31e_critic_information_components.csv"
)

WEIGHTS_PATH = (
    PRIORITY_DIR
    / "31e_critic_weights.csv"
)

AUDIT_PATH = (
    REPORT_DIR
    / "31e_critic_weighting_audit.csv"
)

METHOD_PATH = (
    REPORT_DIR
    / "31e_critic_weighting_methodology.json"
)

REPORT_PATH = (
    REPORT_DIR
    / "31e_critic_weighting_report.txt"
)

HASH_PATH = (
    REPORT_DIR
    / "31e_critic_weighting_hashes.csv"
)


# =============================================================================
# 1. HELPERS
# =============================================================================

def header(title):
    print("\n" + "=" * 132)
    print(title)
    print("=" * 132)


def require(path, label):
    if not path.exists():
        raise FileNotFoundError(
            f"{label} not found:\n{path}"
        )


def require_columns(df, cols, label):
    missing = [
        c
        for c in cols
        if c not in df.columns
    ]

    if missing:
        raise RuntimeError(
            f"{label}: missing columns {missing}"
        )


def require_all_pass(path, label):
    require(path, label)

    df = pd.read_csv(
        path,
        low_memory=False,
    )

    status_col = next(
        (
            c
            for c in df.columns
            if str(c).strip().lower() == "status"
        ),
        None,
    )

    if status_col is None:
        raise RuntimeError(
            f"{label}: no status column"
        )

    status = (
        df[status_col]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    if not status.eq("PASS").all():
        raise RuntimeError(
            f"{label} is not fully PASS"
        )


def sha256_file(path):
    h = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(chunk)

    return h.hexdigest()


def add_audit(rows, check, passed, details):
    rows.append(
        {
            "check": check,
            "status": (
                "PASS"
                if passed
                else "FAIL"
            ),
            "details": details,
        }
    )


# =============================================================================
# 2. VERIFY STEP 31D
# =============================================================================

header(
    "STEP 31E - CRITIC OBJECTIVE WEIGHTING"
)

require_all_pass(
    STEP31D_AUDIT,
    "Step 31D audit",
)

for path, label in [
    (
        STEP31D_NORMALIZED,
        "Step 31D normalized decision matrix",
    ),
    (
        STEP31D_METADATA,
        "Step 31D criterion metadata",
    ),
    (
        STEP31D_PARAMS,
        "Step 31D normalization parameters",
    ),
]:
    require(
        path,
        label,
    )

upstream_hash_before = {
    STEP31D_AUDIT.name:
        sha256_file(STEP31D_AUDIT),

    STEP31D_NORMALIZED.name:
        sha256_file(STEP31D_NORMALIZED),

    STEP31D_METADATA.name:
        sha256_file(STEP31D_METADATA),

    STEP31D_PARAMS.name:
        sha256_file(STEP31D_PARAMS),
}

matrix = pd.read_csv(
    STEP31D_NORMALIZED,
    low_memory=False,
)

metadata = pd.read_csv(
    STEP31D_METADATA,
    low_memory=False,
)

params = pd.read_csv(
    STEP31D_PARAMS,
    low_memory=False,
)

require_columns(
    matrix,
    ["grid_id"],
    "Step 31D normalized matrix",
)

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

if len(metadata) != EXPECTED_CRITERIA:
    raise RuntimeError(
        f"Metadata rows={len(metadata)}, "
        f"expected={EXPECTED_CRITERIA}"
    )

metadata = (
    metadata
    .sort_values(
        "criterion_order"
    )
    .reset_index(
        drop=True
    )
)

fields = (
    metadata[
        "final_field_name"
    ]
    .astype(str)
    .tolist()
)

require_columns(
    matrix,
    fields,
    "Step 31D normalized matrix",
)

if len(matrix) != EXPECTED_GRIDS:
    raise RuntimeError(
        f"Matrix rows={len(matrix)}, "
        f"expected={EXPECTED_GRIDS}"
    )

if (
    matrix[
        "grid_id"
    ].astype(str).nunique()
    != EXPECTED_GRIDS
):
    raise RuntimeError(
        "Normalized matrix does not contain "
        "2,194 unique grids."
    )

if matrix[
    "grid_id"
].astype(str).duplicated().any():
    raise RuntimeError(
        "Duplicate grid_id values found."
    )

for col in fields:
    matrix[col] = pd.to_numeric(
        matrix[col],
        errors="coerce",
    )

    if matrix[col].isna().any():
        raise RuntimeError(
            f"{col}: missing/non-numeric values."
        )

    arr = matrix[col].to_numpy(
        dtype=float
    )

    if not np.isfinite(
        arr
    ).all():
        raise RuntimeError(
            f"{col}: non-finite values."
        )

    if (
        matrix[col].min()
        < -1e-12
        or
        matrix[col].max()
        > 1.0 + 1e-12
    ):
        raise RuntimeError(
            f"{col}: values outside [0,1]."
        )

    if matrix[col].nunique() <= 1:
        raise RuntimeError(
            f"{col}: no variation."
        )

print(
    "PASS: Step 31D verified and normalized "
    "2,194 x 6 matrix loaded."
)


# =============================================================================
# 3. PEARSON CORRELATION MATRIX
# =============================================================================

header(
    "3. CRITIC PEARSON INTER-CRITERIA CORRELATION"
)

X = matrix[
    fields
].astype(float)

corr = X.corr(
    method="pearson"
)

if corr.isna().any().any():
    raise RuntimeError(
        "CRITIC Pearson matrix contains NaN."
    )

corr_values = corr.to_numpy(
    dtype=float
)

if not np.isfinite(
    corr_values
).all():
    raise RuntimeError(
        "CRITIC correlation matrix contains non-finite values."
    )

# Save with criterion codes + names for readability.
field_to_code = dict(
    zip(
        metadata[
            "final_field_name"
        ].astype(str),
        metadata[
            "criterion_code"
        ].astype(str),
    )
)

display_labels = [
    (
        f"{field_to_code[field]}__{field}"
    )
    for field in fields
]

corr_out = corr.copy()
corr_out.index = display_labels
corr_out.columns = display_labels

corr_out.index.name = (
    "criterion"
)

corr_out.to_csv(
    CORR_PATH,
)

print(
    corr_out.round(
        6
    ).to_string()
)


# =============================================================================
# 4. CRITIC INFORMATION CONTENT
# =============================================================================

header(
    "4. CRITIC INFORMATION CONTENT"
)

# Population standard deviation across the 2,194 decision alternatives.
std_population = X.std(
    axis=0,
    ddof=0,
)

# Standard CRITIC conflict term:
# sum over all k of (1 - r_jk).
# The self term contributes 0 because r_jj = 1.
conflict_sum = (
    1.0 - corr
).sum(
    axis=1
)

information = (
    std_population
    * conflict_sum
)

if (
    information
    < -1e-12
).any():
    raise RuntimeError(
        "Negative CRITIC information content found."
    )

information = information.clip(
    lower=0.0
)

information_total = float(
    information.sum()
)

if (
    not np.isfinite(
        information_total
    )
    or
    information_total <= 0
):
    raise RuntimeError(
        "Invalid total CRITIC information content."
    )

weights = (
    information
    / information_total
)

component_rows = []

for _, meta in metadata.iterrows():
    field = str(
        meta[
            "final_field_name"
        ]
    )

    component_rows.append(
        {
            "criterion_order":
                int(
                    meta[
                        "criterion_order"
                    ]
                ),

            "criterion_code":
                meta[
                    "criterion_code"
                ],

            "decision_dimension":
                meta[
                    "decision_dimension"
                ],

            "selected_candidate_id":
                meta[
                    "selected_candidate_id"
                ],

            "final_indicator_name":
                meta[
                    "final_indicator_name"
                ],

            "final_field_name":
                field,

            "std_population":
                float(
                    std_population[
                        field
                    ]
                ),

            "conflict_sum":
                float(
                    conflict_sum[
                        field
                    ]
                ),

            "critic_information":
                float(
                    information[
                        field
                    ]
                ),

            "critic_weight":
                float(
                    weights[
                        field
                    ]
                ),

            "critic_weight_pct":
                float(
                    100.0
                    * weights[
                        field
                    ]
                ),
        }
    )

components = pd.DataFrame(
    component_rows
)

components.to_csv(
    COMPONENT_PATH,
    index=False,
)

weights_out = (
    components[
        [
            "criterion_order",
            "criterion_code",
            "decision_dimension",
            "selected_candidate_id",
            "final_indicator_name",
            "final_field_name",
            "critic_weight",
            "critic_weight_pct",
        ]
    ]
    .copy()
)

weights_out[
    "weight_rank"
] = (
    weights_out[
        "critic_weight"
    ]
    .rank(
        method="min",
        ascending=False,
    )
    .astype(int)
)

weights_out.to_csv(
    WEIGHTS_PATH,
    index=False,
)

print(
    components[
        [
            "criterion_code",
            "decision_dimension",
            "std_population",
            "conflict_sum",
            "critic_information",
            "critic_weight",
            "critic_weight_pct",
        ]
    ].to_string(
        index=False
    )
)

print(
    f"\nCRITIC weight sum: "
    f"{weights.sum():.12f}"
)


# =============================================================================
# 5. DIAGNOSTICS
# =============================================================================

header(
    "5. CRITIC DIAGNOSTICS"
)

# Highest absolute off-diagonal correlation.
abs_corr = corr.abs().copy()

np.fill_diagonal(
    abs_corr.values,
    np.nan,
)

max_abs_corr = float(
    np.nanmax(
        abs_corr.to_numpy(
            dtype=float
        )
    )
)

max_idx = np.unravel_index(
    np.nanargmax(
        abs_corr.to_numpy(
            dtype=float
        )
    ),
    abs_corr.shape,
)

max_pair = (
    fields[
        max_idx[0]
    ],
    fields[
        max_idx[1]
    ],
)

max_pair_r = float(
    corr.loc[
        max_pair[0],
        max_pair[1],
    ]
)

min_weight = float(
    weights.min()
)

max_weight = float(
    weights.max()
)

print(
    f"Highest absolute inter-criterion correlation: "
    f"|r|={max_abs_corr:.6f}"
)

print(
    f"Pair: {field_to_code[max_pair[0]]} vs "
    f"{field_to_code[max_pair[1]]} "
    f"(r={max_pair_r:.6f})"
)

print(
    f"Smallest CRITIC weight: {min_weight:.6f}"
)

print(
    f"Largest CRITIC weight : {max_weight:.6f}"
)

print(
    "\nNote: CRITIC does not remove a highly correlated criterion here; "
    "correlation reduces its unique-information contribution through "
    "the conflict term."
)


# =============================================================================
# 6. STRICT STEP 31E AUDIT
# =============================================================================

header(
    "6. STRICT STEP 31E AUDIT"
)

audit = []

add_audit(
    audit,
    "step31d_audit",
    True,
    "Step 31D audit verified fully PASS.",
)

add_audit(
    audit,
    "grid_count",
    (
        len(matrix)
        == EXPECTED_GRIDS
        and
        matrix[
            "grid_id"
        ].astype(str).nunique()
        == EXPECTED_GRIDS
    ),
    (
        f"rows={len(matrix)}, "
        f"unique_grids="
        f"{matrix['grid_id'].astype(str).nunique()}, "
        f"expected={EXPECTED_GRIDS}"
    ),
)

add_audit(
    audit,
    "criterion_count",
    (
        len(fields)
        == EXPECTED_CRITERIA
    ),
    (
        f"actual={len(fields)}, "
        f"expected={EXPECTED_CRITERIA}"
    ),
)

add_audit(
    audit,
    "normalized_matrix_complete",
    (
        X.isna().sum().sum()
        == 0
        and
        np.isfinite(
            X.to_numpy(
                dtype=float
            )
        ).all()
    ),
    "No missing or non-finite normalized criterion values.",
)

add_audit(
    audit,
    "normalized_values_within_0_1",
    bool(
        (
            X.to_numpy(
                dtype=float
            )
            >= -1e-12
        ).all()
        and
        (
            X.to_numpy(
                dtype=float
            )
            <= 1.0 + 1e-12
        ).all()
    ),
    "All Step31D normalized values lie inside [0,1].",
)

add_audit(
    audit,
    "correlation_matrix_finite",
    bool(
        np.isfinite(
            corr_values
        ).all()
    ),
    "Pearson correlation matrix has only finite values.",
)

add_audit(
    audit,
    "correlation_matrix_symmetric",
    bool(
        np.allclose(
            corr_values,
            corr_values.T,
            atol=1e-12,
            rtol=0.0,
        )
    ),
    "Pearson correlation matrix is symmetric.",
)

add_audit(
    audit,
    "correlation_diagonal_one",
    bool(
        np.allclose(
            np.diag(
                corr_values
            ),
            1.0,
            atol=1e-12,
            rtol=0.0,
        )
    ),
    "All correlation diagonal values equal 1.",
)

add_audit(
    audit,
    "all_standard_deviations_positive",
    bool(
        (
            std_population
            > 0
        ).all()
    ),
    "All six normalized criteria have positive population standard deviation.",
)

add_audit(
    audit,
    "all_conflict_sums_nonnegative",
    bool(
        (
            conflict_sum
            >= -1e-12
        ).all()
    ),
    "All CRITIC conflict sums are non-negative.",
)

add_audit(
    audit,
    "all_information_values_nonnegative",
    bool(
        (
            information
            >= -1e-12
        ).all()
    ),
    "All CRITIC information-content values are non-negative.",
)

add_audit(
    audit,
    "all_weights_nonnegative",
    bool(
        (
            weights
            >= -1e-12
        ).all()
    ),
    "All CRITIC weights are non-negative.",
)

add_audit(
    audit,
    "weight_sum_one",
    bool(
        np.isclose(
            float(
                weights.sum()
            ),
            1.0,
            atol=1e-12,
            rtol=0.0,
        )
    ),
    (
        f"weight_sum="
        f"{float(weights.sum()):.12f}"
    ),
)

add_audit(
    audit,
    "all_six_weights_positive",
    bool(
        (
            weights
            > 0
        ).all()
    ),
    "Every frozen criterion receives a positive CRITIC weight.",
)

add_audit(
    audit,
    "no_manual_weights_mixed",
    True,
    "Weights are derived only from normalized variability and Pearson inter-criterion correlation.",
)

add_audit(
    audit,
    "no_new_indicator_selection",
    True,
    "Exactly the six Step31C-frozen indicators are retained.",
)

add_audit(
    audit,
    "no_topsis_ranking",
    True,
    "TOPSIS is deferred to Step32.",
)

add_audit(
    audit,
    "no_model_retraining",
    True,
    "No model fitting/retraining performed.",
)

upstream_hash_after = {
    STEP31D_AUDIT.name:
        sha256_file(STEP31D_AUDIT),

    STEP31D_NORMALIZED.name:
        sha256_file(STEP31D_NORMALIZED),

    STEP31D_METADATA.name:
        sha256_file(STEP31D_METADATA),

    STEP31D_PARAMS.name:
        sha256_file(STEP31D_PARAMS),
}

add_audit(
    audit,
    "upstream_step31d_files_unchanged",
    (
        upstream_hash_before
        == upstream_hash_after
    ),
    "Step31D input hashes are unchanged before vs after Step31E.",
)

audit_df = pd.DataFrame(
    audit
)

audit_df.to_csv(
    AUDIT_PATH,
    index=False,
)

print(
    audit_df.to_string(
        index=False
    )
)

if not audit_df[
    "status"
].eq("PASS").all():
    raise RuntimeError(
        "Step 31E strict audit FAILED. "
        "Do not proceed to Step 32."
    )


# =============================================================================
# 7. METHODOLOGY + REPORT
# =============================================================================

methodology = {
    "step": "31E",
    "title": "CRITIC Objective Weighting",
    "grid_count": EXPECTED_GRIDS,
    "criterion_count": EXPECTED_CRITERIA,
    "input_matrix": (
        "Step31D min-max normalized priority-oriented matrix"
    ),
    "correlation": {
        "type": "Pearson",
        "reason": (
            "Standard CRITIC formulation uses inter-criterion "
            "linear correlation to quantify conflict/redundancy."
        ),
    },
    "dispersion": {
        "measure": "population standard deviation",
        "ddof": 0,
    },
    "formula": {
        "conflict_j": "sum_k(1 - r_jk)",
        "information_j": "sigma_j * conflict_j",
        "weight_j": "information_j / sum_j(information_j)",
    },
    "interpretation": (
        "Higher weight means greater objective information contribution "
        "through stronger contrast and/or lower redundancy with other criteria."
    ),
    "manual_weighting_used": False,
    "criterion_removed": False,
    "topsis_performed": False,
    "model_retrained": False,
    "next": "Step32 TOPSIS priority ranking",
}

with METHOD_PATH.open(
    "w",
    encoding="utf-8",
) as f:
    json.dump(
        methodology,
        f,
        indent=2,
        ensure_ascii=False,
    )

report_lines = [
    "STEP 31E - CRITIC OBJECTIVE WEIGHTING",
    "=" * 52,
    "",
    "STATUS: COMPLETED",
    "",
    f"Grids    : {EXPECTED_GRIDS}",
    f"Criteria : {EXPECTED_CRITERIA}",
    "",
    "FORMULA:",
    "conflict_j = sum_k(1 - Pearson_r_jk)",
    "information_j = population_std_j * conflict_j",
    "weight_j = information_j / sum(information)",
    "",
    "FINAL CRITIC WEIGHTS:",
]

for _, row in weights_out.sort_values(
    "criterion_order"
).iterrows():
    report_lines.append(
        (
            f"{row['criterion_code']} | "
            f"{row['decision_dimension']} | "
            f"weight={row['critic_weight']:.8f} | "
            f"{row['critic_weight_pct']:.4f}%"
        )
    )

report_lines += [
    "",
    f"Weight sum: {weights.sum():.12f}",
    (
        "Highest absolute inter-criterion correlation: "
        f"{max_abs_corr:.6f}"
    ),
    "",
    "IMPORTANT:",
    "- weights are objective CRITIC weights",
    "- no expert/manual weights were mixed in",
    "- no criterion was removed",
    "- no TOPSIS ranking was performed",
    "- no model retraining was performed",
    "",
    "NEXT:",
    "Step32 -> TOPSIS priority ranking.",
]

REPORT_PATH.write_text(
    "\n".join(
        report_lines
    ),
    encoding="utf-8",
)


# =============================================================================
# 8. HASHES
# =============================================================================

hash_targets = [
    CORR_PATH,
    COMPONENT_PATH,
    WEIGHTS_PATH,
    AUDIT_PATH,
    METHOD_PATH,
    REPORT_PATH,
]

hash_df = pd.DataFrame(
    [
        {
            "file": p.name,
            "sha256": sha256_file(
                p
            ),
        }
        for p in hash_targets
    ]
)

hash_df.to_csv(
    HASH_PATH,
    index=False,
)


# =============================================================================
# 9. FINAL
# =============================================================================

header(
    "STEP 31E COMPLETED SUCCESSFULLY"
)

print(
    "CRITIC weights calculated for all six frozen criteria."
)

print(
    f"Weight sum: {weights.sum():.12f}"
)

print("\nFinal weights:")
print(
    weights_out
    .sort_values(
        "criterion_order"
    )[
        [
            "criterion_code",
            "decision_dimension",
            "critic_weight",
            "critic_weight_pct",
            "weight_rank",
        ]
    ]
    .to_string(
        index=False
    )
)

print("\nSaved:")
for p in [
    CORR_PATH,
    COMPONENT_PATH,
    WEIGHTS_PATH,
    AUDIT_PATH,
    METHOD_PATH,
    REPORT_PATH,
    HASH_PATH,
]:
    print(
        f"- {p}"
    )

print(
    "\nNo TOPSIS ranking performed."
)
print(
    "No model retraining performed."
)
print(
    "NEXT: review Step31E outputs before Step32."
)
