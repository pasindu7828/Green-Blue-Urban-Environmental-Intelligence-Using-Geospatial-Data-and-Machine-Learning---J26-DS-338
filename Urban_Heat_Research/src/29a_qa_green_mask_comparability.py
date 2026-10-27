"""
STEP 29A-QA - GREEN-MASK COMPARABILITY & RESPONSE VALIDITY AUDIT
Urban Heat Research - Kaduwela, Sri Lanka

WHY THIS STEP?
--------------
Step 29A V3/V3S showed a robust positive within-grid association between
green_mask and LST. Before using green_mask as an intervention variable, we
must determine whether:

1) green_mask is longitudinally comparable across years;
2) green_mask behaves consistently with NDVI, the vegetation index from which
   the Otsu green-space mask was derived;
3) the positive green coefficient appears only after adding controls
   (suppression / sign-reversal diagnostic);
4) NDVI and EVI show the same or different within-grid response pattern;
5) the result is sensitive to 2017;
6) any local GEE/source files reveal whether Otsu thresholds were fixed or
   recalculated by year.

IMPORTANT
---------
This is a QA / validity audit. It does NOT:
- alter the historical dataset;
- alter Step-28 forecasts;
- retrain Step-22B LightGBM;
- force a cooling sign;
- declare causality;
- infer Otsu threshold provenance when the source code/log is unavailable.

If Otsu threshold provenance cannot be found, the script explicitly reports
that longitudinal comparability of green_mask cannot be proven from the
aggregated CSV alone.
"""

from pathlib import Path
import hashlib
import json
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy.stats import pearsonr, spearmanr


# =============================================================================
# SETTINGS
# =============================================================================

GRID_COL = "grid_id"
YEAR_COL = "year"
TARGET = "LST_C"

ALL_YEARS = list(range(2015, 2026))
NO_2017_YEARS = [year for year in ALL_YEARS if year != 2017]

EXPECTED_GRIDS = 2194
EXPECTED_ROWS = EXPECTED_GRIDS * len(ALL_YEARS)

GREEN = "green_mask"
NDVI = "NDVI"
EVI = "EVI"

BASE_CONTROLS = [
    "NDBI",
    "NDWI",
    "Albedo",
    "NightLights",
]

NUMERIC_TOL = 1e-12

EXPECTED_HASHES = {
    "kaduwela_preprocessed_base.csv":
        "0e030b2ec0b5c4080cd4226bec798189a178aa45efd9962d19f38dc9cb170f46",
    "29a_v3_twfe_primary_coefficients.csv":
        "c6813b6c3902f34b679368ee3a1daa15dd4c5270a24811ffc3436daea39eedb3",
    "29a_v3s_2017_sensitivity_audit.csv":
        "3317c1f3ee44e90157b859ad05421f1dcaa7f015a491b12ceb15d53d3a83a4be",
}

V3_GREEN_BETA_PER_10PP = 0.1971988411609
V3_BETA_REPRO_TOL = 1e-9


# =============================================================================
# PATHS
# =============================================================================

ROOT = Path(__file__).resolve().parents[1]

BASE_PATH = (
    ROOT / "data" / "processed"
    / "kaduwela_preprocessed_base.csv"
)

V3_COEF_PATH = (
    ROOT / "outputs" / "reports"
    / "29a_v3_twfe_primary_coefficients.csv"
)

V3S_AUDIT_PATH = (
    ROOT / "outputs" / "reports"
    / "29a_v3s_2017_sensitivity_audit.csv"
)

REPORT_DIR = ROOT / "outputs" / "reports"
FIGURE_DIR = ROOT / "outputs" / "figures"

for folder in [
    REPORT_DIR,
    FIGURE_DIR,
]:
    folder.mkdir(parents=True, exist_ok=True)


YEARLY_RELATION_PATH = (
    REPORT_DIR
    / "29a_qa_yearly_green_ndvi_relationships.csv"
)

DELTA_RELATION_PATH = (
    REPORT_DIR
    / "29a_qa_within_grid_delta_relationships.csv"
)

WITHIN_CORRELATION_PATH = (
    REPORT_DIR
    / "29a_qa_two_way_demeaned_correlations.csv"
)

GREEN_NESTED_PATH = (
    REPORT_DIR
    / "29a_qa_green_nested_twfe.csv"
)

NDVI_NESTED_PATH = (
    REPORT_DIR
    / "29a_qa_ndvi_nested_twfe.csv"
)

VEGETATION_COMPARE_PATH = (
    REPORT_DIR
    / "29a_qa_vegetation_indicator_twfe_comparison.csv"
)

PROVENANCE_SEARCH_PATH = (
    REPORT_DIR
    / "29a_qa_otsu_provenance_search.csv"
)

QA_SUMMARY_PATH = (
    REPORT_DIR
    / "29a_qa_green_mask_validity_summary.csv"
)

AUDIT_PATH = (
    REPORT_DIR
    / "29a_qa_green_mask_comparability_audit.csv"
)

METHOD_PATH = (
    REPORT_DIR
    / "29a_qa_green_mask_comparability_methodology.json"
)

REPORT_PATH = (
    REPORT_DIR
    / "29a_qa_green_mask_comparability_report.txt"
)

HASH_PATH = (
    REPORT_DIR
    / "29a_qa_green_mask_comparability_hashes.csv"
)

YEARLY_FIGURE_PATH = (
    FIGURE_DIR
    / "29a_qa_yearly_green_ndvi_correlations.png"
)

NESTED_FIGURE_PATH = (
    FIGURE_DIR
    / "29a_qa_nested_twfe_green_coefficients.png"
)

VEGETATION_FIGURE_PATH = (
    FIGURE_DIR
    / "29a_qa_vegetation_indicator_comparison.png"
)


# =============================================================================
# HELPERS
# =============================================================================

def section(title):
    print("\n" + "=" * 132)
    print(title)
    print("=" * 132)


def sha256_file(path):
    h = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)

    return h.hexdigest()


def verify_hash(path, expected, label):
    if not path.exists():
        raise FileNotFoundError(
            f"Missing required file: {path}"
        )

    actual = sha256_file(path)

    if actual != expected:
        raise RuntimeError(
            f"{label} hash mismatch.\n"
            f"Expected: {expected}\n"
            f"Actual  : {actual}"
        )

    print(
        f"PASS hash: {label} = {actual}"
    )

    return actual


def require_all_pass(path, label):
    audit = pd.read_csv(path)

    if "status" not in audit.columns:
        raise RuntimeError(
            f"{label} has no status column."
        )

    bad = audit[
        audit["status"].astype(str).str.upper()
        != "PASS"
    ]

    if not bad.empty:
        print(
            bad.to_string(index=False)
        )

        raise RuntimeError(
            f"{label} is not fully PASS."
        )


def check_balanced_panel(df, years):
    if df[
        GRID_COL
    ].nunique() != EXPECTED_GRIDS:
        raise RuntimeError(
            f"Unique grids={df[GRID_COL].nunique()}, "
            f"expected={EXPECTED_GRIDS}"
        )

    expected_rows = (
        EXPECTED_GRIDS
        * len(years)
    )

    if len(df) != expected_rows:
        raise RuntimeError(
            f"Rows={len(df)}, expected={expected_rows}"
        )

    counts = (
        df.groupby(
            GRID_COL
        )[
            YEAR_COL
        ]
        .nunique()
    )

    if not (
        counts == len(years)
    ).all():
        raise RuntimeError(
            "Panel is not balanced."
        )


def two_way_demean(df, columns):
    transformed = pd.DataFrame(
        index=df.index
    )

    for column in columns:
        values = pd.to_numeric(
            df[column],
            errors="raise",
        ).astype(float)

        entity_mean = (
            values.groupby(
                df[GRID_COL]
            )
            .transform("mean")
        )

        year_mean = (
            values.groupby(
                df[YEAR_COL]
            )
            .transform("mean")
        )

        transformed[column] = (
            values
            - entity_mean
            - year_mean
            + float(values.mean())
        )

    return transformed


def fit_twfe(panel, years, treatment, controls, model_name):
    sub = (
        panel[
            panel[YEAR_COL].isin(years)
        ]
        .copy()
        .sort_values(
            [GRID_COL, YEAR_COL]
        )
        .reset_index(drop=True)
    )

    check_balanced_panel(
        sub,
        years,
    )

    cols = [
        TARGET,
        treatment,
        *controls,
    ]

    transformed = two_way_demean(
        sub,
        cols,
    )

    y = transformed[
        TARGET
    ].to_numpy(dtype=float)

    X = transformed[
        [
            treatment,
            *controls,
        ]
    ].copy()

    # No near-zero within variance.
    variances = X.var(ddof=0)

    bad = variances[
        variances <= NUMERIC_TOL
    ]

    if not bad.empty:
        raise RuntimeError(
            f"{model_name}: near-zero within variance:\n"
            + bad.to_string()
        )

    rank = int(
        np.linalg.matrix_rank(
            X.to_numpy(dtype=float)
        )
    )

    if rank != X.shape[1]:
        raise RuntimeError(
            f"{model_name}: rank={rank}, cols={X.shape[1]}"
        )

    fit = sm.OLS(
        y,
        X,
        hasconst=False,
    ).fit(
        cov_type="cluster",
        cov_kwds={
            "groups":
                sub[GRID_COL]
                .astype(str)
                .to_numpy(),
            "use_correction":
                True,
        },
    )

    ci = fit.conf_int(
        alpha=0.05
    ).loc[treatment]

    beta = float(
        fit.params[treatment]
    )

    lower = float(ci[0])
    upper = float(ci[1])

    # For fraction-like vegetation metrics, also show +0.10 response.
    response_per_010 = beta * 0.10

    return {
        "model_name":
            model_name,
        "treatment":
            treatment,
        "controls":
            controls,
        "years":
            years,
        "rows":
            int(fit.nobs),
        "beta":
            beta,
        "response_per_0_10":
            response_per_010,
        "se_clustered":
            float(fit.bse[treatment]),
        "p_value":
            float(fit.pvalues[treatment]),
        "ci95_lower":
            lower,
        "ci95_upper":
            upper,
        "ci95_lower_per_0_10":
            lower * 0.10,
        "ci95_upper_per_0_10":
            upper * 0.10,
        "within_r2":
            float(fit.rsquared),
        "fit":
            fit,
    }


def safe_corr(x, y, method):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    mask = (
        np.isfinite(x)
        & np.isfinite(y)
    )

    x = x[mask]
    y = y[mask]

    if len(x) < 3:
        return np.nan, np.nan

    if np.std(x) <= NUMERIC_TOL:
        return np.nan, np.nan

    if np.std(y) <= NUMERIC_TOL:
        return np.nan, np.nan

    if method == "pearson":
        r, p = pearsonr(
            x,
            y,
        )
    elif method == "spearman":
        r, p = spearmanr(
            x,
            y,
        )
    else:
        raise ValueError(method)

    return float(r), float(p)


def sign_label(value):
    if value > 0:
        return "POSITIVE"
    if value < 0:
        return "NEGATIVE"
    return "ZERO"


def ci_position(lower, upper):
    if lower > 0:
        return "ENTIRELY_POSITIVE"
    if upper < 0:
        return "ENTIRELY_NEGATIVE"
    return "INCLUDES_ZERO"


def scan_project_for_otsu_provenance(root):
    """
    Search reasonably small text/source files for evidence about how the Otsu
    threshold was generated. This is evidence discovery only; the script does
    NOT infer fixed-vs-yearly behavior from ambiguous text.
    """

    allowed_suffixes = {
        ".py",
        ".js",
        ".txt",
        ".md",
        ".json",
        ".csv",
    }

    excluded_dir_names = {
        ".venv",
        "venv",
        "__pycache__",
        ".git",
        "node_modules",
        "models",
        "figures",
        "maps",
        "scenarios",
    }

    terms = [
        "otsu",
        "green_mask",
        "threshold",
    ]

    hits = []

    for path in root.rglob("*"):
        if not path.is_file():
            continue

        if any(
            part in excluded_dir_names
            for part in path.parts
        ):
            continue

        if path.suffix.lower() not in allowed_suffixes:
            continue

        try:
            if path.stat().st_size > 2_000_000:
                continue
        except OSError:
            continue

        # Do not treat QA outputs themselves as provenance evidence.
        if "29a_qa_" in path.name.lower():
            continue

        try:
            text = path.read_text(
                encoding="utf-8",
                errors="ignore",
            )
        except Exception:
            continue

        low = text.lower()

        if "otsu" not in low:
            continue

        lines = text.splitlines()

        for idx, line in enumerate(lines, start=1):
            low_line = line.lower()

            if (
                "otsu" in low_line
                or (
                    "green_mask" in low_line
                    and "threshold" in low_line
                )
            ):
                context_start = max(
                    0,
                    idx - 3,
                )

                context_end = min(
                    len(lines),
                    idx + 2,
                )

                context = " | ".join(
                    item.strip()
                    for item in lines[
                        context_start:
                        context_end
                    ]
                    if item.strip()
                )

                hits.append(
                    {
                        "file":
                            str(
                                path.relative_to(
                                    root
                                )
                            ),
                        "line":
                            idx,
                        "matched_line":
                            line.strip()[:500],
                        "context":
                            context[:1500],
                    }
                )

    return pd.DataFrame(hits)


# =============================================================================
# 0. VERIFY INPUTS
# =============================================================================

section(
    "STEP 29A-QA - GREEN-MASK COMPARABILITY & RESPONSE VALIDITY AUDIT"
)

verify_hash(
    BASE_PATH,
    EXPECTED_HASHES[
        "kaduwela_preprocessed_base.csv"
    ],
    "Canonical historical base data",
)

verify_hash(
    V3_COEF_PATH,
    EXPECTED_HASHES[
        "29a_v3_twfe_primary_coefficients.csv"
    ],
    "Step-29A V3 coefficient table",
)

verify_hash(
    V3S_AUDIT_PATH,
    EXPECTED_HASHES[
        "29a_v3s_2017_sensitivity_audit.csv"
    ],
    "Step-29A V3S strict audit",
)

require_all_pass(
    V3S_AUDIT_PATH,
    "Step-29A V3S audit",
)

print(
    "\nPASS: Step-29A V3S audit is fully PASS."
)


# =============================================================================
# 1. LOAD + VALIDATE PANEL
# =============================================================================

section(
    "1. LOAD + VALIDATE 2015-2025 PANEL"
)

base = pd.read_csv(
    BASE_PATH,
    dtype={
        GRID_COL: str,
        "area_name": str,
    },
)

base[
    YEAR_COL
] = pd.to_numeric(
    base[YEAR_COL],
    errors="raise",
).astype(int)

required = [
    GRID_COL,
    YEAR_COL,
    TARGET,
    GREEN,
    NDVI,
    EVI,
    *BASE_CONTROLS,
]

missing = [
    column
    for column in required
    if column not in base.columns
]

if missing:
    raise RuntimeError(
        "Missing QA columns:\n"
        + str(missing)
    )

panel = (
    base[
        base[YEAR_COL].isin(
            ALL_YEARS
        )
    ][required]
    .copy()
    .sort_values(
        [GRID_COL, YEAR_COL]
    )
    .reset_index(drop=True)
)

for column in [
    TARGET,
    GREEN,
    NDVI,
    EVI,
    *BASE_CONTROLS,
]:
    panel[column] = pd.to_numeric(
        panel[column],
        errors="raise",
    )

check_balanced_panel(
    panel,
    ALL_YEARS,
)

if panel[
    [
        TARGET,
        GREEN,
        NDVI,
        EVI,
        *BASE_CONTROLS,
    ]
].isna().any().any():
    raise RuntimeError(
        "QA panel contains missing values."
    )

if not np.isfinite(
    panel[
        [
            TARGET,
            GREEN,
            NDVI,
            EVI,
            *BASE_CONTROLS,
        ]
    ].to_numpy(dtype=float)
).all():
    raise RuntimeError(
        "QA panel contains non-finite values."
    )

print(
    f"Rows         : {len(panel):,}"
)

print(
    f"Unique grids : {panel[GRID_COL].nunique():,}"
)

print(
    f"Years        : {ALL_YEARS}"
)


# =============================================================================
# 2. OTSU THRESHOLD PROVENANCE SEARCH
# =============================================================================

section(
    "2. SEARCH LOCAL PROJECT FILES FOR OTSU THRESHOLD PROVENANCE"
)

provenance_hits = scan_project_for_otsu_provenance(
    ROOT
)

if provenance_hits.empty:
    provenance_hits = pd.DataFrame(
        columns=[
            "file",
            "line",
            "matched_line",
            "context",
        ]
    )

provenance_hits.to_csv(
    PROVENANCE_SEARCH_PATH,
    index=False,
)

if provenance_hits.empty:
    print(
        "No local text/source evidence containing 'Otsu' was found."
    )
else:
    print(
        provenance_hits.head(
            40
        ).to_string(
            index=False
        )
    )

print(
    "\nIMPORTANT:"
)

print(
    "The aggregated kaduwela_preprocessed_base.csv does not contain an "
    "Otsu-threshold-by-year field."
)

print(
    "Therefore fixed-vs-year-specific threshold provenance cannot be proven "
    "from the modeling CSV alone."
)

print(
    "Any discovered source hits must be manually reviewed before claiming "
    "how thresholds were generated."
)


# =============================================================================
# 3. YEARLY GREEN_MASK <-> NDVI/EVI RELATIONSHIPS
# =============================================================================

section(
    "3. YEARLY GREEN_MASK RELATIONSHIP WITH NDVI / EVI / LST"
)

yearly_rows = []

for year in ALL_YEARS:
    sub = panel[
        panel[YEAR_COL] == year
    ]

    green_ndvi_p, green_ndvi_pp = safe_corr(
        sub[GREEN],
        sub[NDVI],
        "pearson",
    )

    green_ndvi_s, green_ndvi_sp = safe_corr(
        sub[GREEN],
        sub[NDVI],
        "spearman",
    )

    green_evi_p, green_evi_pp = safe_corr(
        sub[GREEN],
        sub[EVI],
        "pearson",
    )

    green_lst_p, green_lst_pp = safe_corr(
        sub[GREEN],
        sub[TARGET],
        "pearson",
    )

    ndvi_lst_p, ndvi_lst_pp = safe_corr(
        sub[NDVI],
        sub[TARGET],
        "pearson",
    )

    yearly_rows.append(
        {
            "year":
                year,
            "mean_green_mask":
                float(
                    sub[GREEN].mean()
                ),
            "mean_ndvi":
                float(
                    sub[NDVI].mean()
                ),
            "mean_evi":
                float(
                    sub[EVI].mean()
                ),
            "pearson_green_ndvi":
                green_ndvi_p,
            "pearson_green_ndvi_p":
                green_ndvi_pp,
            "spearman_green_ndvi":
                green_ndvi_s,
            "spearman_green_ndvi_p":
                green_ndvi_sp,
            "pearson_green_evi":
                green_evi_p,
            "pearson_green_evi_p":
                green_evi_pp,
            "pearson_green_lst":
                green_lst_p,
            "pearson_green_lst_p":
                green_lst_pp,
            "pearson_ndvi_lst":
                ndvi_lst_p,
            "pearson_ndvi_lst_p":
                ndvi_lst_pp,
        }
    )

yearly_df = pd.DataFrame(
    yearly_rows
)

yearly_df.to_csv(
    YEARLY_RELATION_PATH,
    index=False,
)

print(
    yearly_df.to_string(
        index=False
    )
)

green_ndvi_all_positive = bool(
    (
        yearly_df[
            "pearson_green_ndvi"
        ]
        > 0
    ).all()
)

green_ndvi_min_corr = float(
    yearly_df[
        "pearson_green_ndvi"
    ].min()
)

green_ndvi_median_corr = float(
    yearly_df[
        "pearson_green_ndvi"
    ].median()
)


# =============================================================================
# 4. WITHIN-GRID YEAR-TO-YEAR DELTA RELATIONSHIPS
# =============================================================================

section(
    "4. WITHIN-GRID YEAR-TO-YEAR DELTA RELATIONSHIPS"
)

delta_panel = (
    panel
    .sort_values(
        [GRID_COL, YEAR_COL]
    )
    .copy()
)

for column in [
    TARGET,
    GREEN,
    NDVI,
    EVI,
    *BASE_CONTROLS,
]:
    delta_panel[
        f"d_{column}"
    ] = (
        delta_panel.groupby(
            GRID_COL
        )[column]
        .diff()
    )

delta_panel[
    "previous_year"
] = (
    delta_panel.groupby(
        GRID_COL
    )[YEAR_COL]
    .shift(1)
)

delta_panel = (
    delta_panel[
        delta_panel[YEAR_COL]
        .between(2016, 2025)
    ]
    .copy()
    .reset_index(drop=True)
)

if not (
    delta_panel[YEAR_COL]
    - delta_panel["previous_year"]
    == 1
).all():
    raise RuntimeError(
        "Non-adjacent year differences detected."
    )

delta_variants = {
    "ALL_TRANSITIONS":
        delta_panel,
    "EXCLUDE_TRANSITIONS_TOUCHING_2017":
        delta_panel[
            ~delta_panel[
                YEAR_COL
            ].isin(
                [2017, 2018]
            )
        ],
}

delta_rows = []

for label, sub in delta_variants.items():
    dgreen_dndvi_p, p1 = safe_corr(
        sub[f"d_{GREEN}"],
        sub[f"d_{NDVI}"],
        "pearson",
    )

    dgreen_dndvi_s, p2 = safe_corr(
        sub[f"d_{GREEN}"],
        sub[f"d_{NDVI}"],
        "spearman",
    )

    dgreen_dlst_p, p3 = safe_corr(
        sub[f"d_{GREEN}"],
        sub[f"d_{TARGET}"],
        "pearson",
    )

    dndvi_dlst_p, p4 = safe_corr(
        sub[f"d_{NDVI}"],
        sub[f"d_{TARGET}"],
        "pearson",
    )

    delta_rows.append(
        {
            "variant":
                label,
            "rows":
                len(sub),
            "pearson_dgreen_dndvi":
                dgreen_dndvi_p,
            "pearson_dgreen_dndvi_p":
                p1,
            "spearman_dgreen_dndvi":
                dgreen_dndvi_s,
            "spearman_dgreen_dndvi_p":
                p2,
            "pearson_dgreen_dlst":
                dgreen_dlst_p,
            "pearson_dgreen_dlst_p":
                p3,
            "pearson_dndvi_dlst":
                dndvi_dlst_p,
            "pearson_dndvi_dlst_p":
                p4,
        }
    )

delta_df = pd.DataFrame(
    delta_rows
)

delta_df.to_csv(
    DELTA_RELATION_PATH,
    index=False,
)

print(
    delta_df.to_string(
        index=False
    )
)


# =============================================================================
# 5. TWO-WAY-DEMEANED CORRELATIONS
# =============================================================================

section(
    "5. TWO-WAY-DEMEANED WITHIN-GRID/YEAR CORRELATIONS"
)

tw = two_way_demean(
    panel,
    [
        TARGET,
        GREEN,
        NDVI,
        EVI,
        *BASE_CONTROLS,
    ],
)

within_pairs = [
    (
        GREEN,
        NDVI,
    ),
    (
        GREEN,
        EVI,
    ),
    (
        GREEN,
        TARGET,
    ),
    (
        NDVI,
        TARGET,
    ),
    (
        EVI,
        TARGET,
    ),
    (
        GREEN,
        "NDBI",
    ),
    (
        GREEN,
        "NDWI",
    ),
    (
        GREEN,
        "Albedo",
    ),
    (
        GREEN,
        "NightLights",
    ),
]

within_rows = []

for a, b in within_pairs:
    rp, pp = safe_corr(
        tw[a],
        tw[b],
        "pearson",
    )

    rs, sp = safe_corr(
        tw[a],
        tw[b],
        "spearman",
    )

    within_rows.append(
        {
            "variable_a":
                a,
            "variable_b":
                b,
            "pearson_two_way_demeaned":
                rp,
            "pearson_p":
                pp,
            "spearman_two_way_demeaned":
                rs,
            "spearman_p":
                sp,
        }
    )

within_df = pd.DataFrame(
    within_rows
)

within_df.to_csv(
    WITHIN_CORRELATION_PATH,
    index=False,
)

print(
    within_df.to_string(
        index=False
    )
)


# =============================================================================
# 6. NESTED TWFE - GREEN MASK
# =============================================================================

section(
    "6. NESTED TWFE DIAGNOSTIC - GREEN_MASK"
)

nested_specs = [
    (
        "G0_GREEN_ONLY",
        [],
    ),
    (
        "G1_GREEN_NDBI",
        [
            "NDBI",
        ],
    ),
    (
        "G2_GREEN_NDBI_NDWI",
        [
            "NDBI",
            "NDWI",
        ],
    ),
    (
        "G3_GREEN_PLUS_ALBEDO",
        [
            "NDBI",
            "NDWI",
            "Albedo",
        ],
    ),
    (
        "G4_GREEN_FULL_V3_CONTROLS",
        [
            "NDBI",
            "NDWI",
            "Albedo",
            "NightLights",
        ],
    ),
]

green_nested_rows = []

for model_name, controls in nested_specs:
    result = fit_twfe(
        panel=panel,
        years=ALL_YEARS,
        treatment=GREEN,
        controls=controls,
        model_name=model_name,
    )

    green_nested_rows.append(
        {
            "model":
                model_name,
            "controls":
                ";".join(controls)
                if controls
                else "NONE",
            "green_beta_c_per_fraction":
                result["beta"],
            "green_beta_c_per_10pp":
                result[
                    "response_per_0_10"
                ],
            "clustered_se":
                result[
                    "se_clustered"
                ],
            "p_value":
                result[
                    "p_value"
                ],
            "ci95_lower_per_10pp":
                result[
                    "ci95_lower_per_0_10"
                ],
            "ci95_upper_per_10pp":
                result[
                    "ci95_upper_per_0_10"
                ],
            "sign":
                sign_label(
                    result["beta"]
                ),
            "ci_position":
                ci_position(
                    result[
                        "ci95_lower"
                    ],
                    result[
                        "ci95_upper"
                    ],
                ),
            "within_r2":
                result[
                    "within_r2"
                ],
        }
    )

green_nested_df = pd.DataFrame(
    green_nested_rows
)

green_nested_df.to_csv(
    GREEN_NESTED_PATH,
    index=False,
)

print(
    green_nested_df.to_string(
        index=False
    )
)

green_only_sign = str(
    green_nested_df.iloc[
        0
    ][
        "sign"
    ]
)

green_full_sign = str(
    green_nested_df.iloc[
        -1
    ][
        "sign"
    ]
)

green_sign_reversal_across_controls = (
    green_only_sign
    != green_full_sign
)


# =============================================================================
# 7. NESTED TWFE - NDVI
# =============================================================================

section(
    "7. NESTED TWFE DIAGNOSTIC - NDVI"
)

ndvi_specs = [
    (
        "N0_NDVI_ONLY",
        [],
    ),
    (
        "N1_NDVI_NDBI",
        [
            "NDBI",
        ],
    ),
    (
        "N2_NDVI_NDBI_NDWI",
        [
            "NDBI",
            "NDWI",
        ],
    ),
    (
        "N3_NDVI_PLUS_ALBEDO",
        [
            "NDBI",
            "NDWI",
            "Albedo",
        ],
    ),
    (
        "N4_NDVI_FULL_CONTROLS",
        [
            "NDBI",
            "NDWI",
            "Albedo",
            "NightLights",
        ],
    ),
]

ndvi_nested_rows = []

for model_name, controls in ndvi_specs:
    result = fit_twfe(
        panel=panel,
        years=ALL_YEARS,
        treatment=NDVI,
        controls=controls,
        model_name=model_name,
    )

    ndvi_nested_rows.append(
        {
            "model":
                model_name,
            "controls":
                ";".join(controls)
                if controls
                else "NONE",
            "ndvi_beta_c_per_1_0":
                result["beta"],
            "ndvi_beta_c_per_0_10":
                result[
                    "response_per_0_10"
                ],
            "clustered_se":
                result[
                    "se_clustered"
                ],
            "p_value":
                result[
                    "p_value"
                ],
            "ci95_lower_per_0_10":
                result[
                    "ci95_lower_per_0_10"
                ],
            "ci95_upper_per_0_10":
                result[
                    "ci95_upper_per_0_10"
                ],
            "sign":
                sign_label(
                    result["beta"]
                ),
            "ci_position":
                ci_position(
                    result[
                        "ci95_lower"
                    ],
                    result[
                        "ci95_upper"
                    ],
                ),
            "within_r2":
                result[
                    "within_r2"
                ],
        }
    )

ndvi_nested_df = pd.DataFrame(
    ndvi_nested_rows
)

ndvi_nested_df.to_csv(
    NDVI_NESTED_PATH,
    index=False,
)

print(
    ndvi_nested_df.to_string(
        index=False
    )
)


# =============================================================================
# 8. FULL-CONTROL VEGETATION INDICATOR COMPARISON
# =============================================================================

section(
    "8. FULL-CONTROL VEGETATION INDICATOR COMPARISON"
)

vegetation_rows = []

for indicator in [
    GREEN,
    NDVI,
    EVI,
]:
    full_result = fit_twfe(
        panel=panel,
        years=ALL_YEARS,
        treatment=indicator,
        controls=BASE_CONTROLS,
        model_name=
            f"{indicator}_FULL_2015_2025",
    )

    no2017_result = fit_twfe(
        panel=panel,
        years=NO_2017_YEARS,
        treatment=indicator,
        controls=BASE_CONTROLS,
        model_name=
            f"{indicator}_FULL_EXCLUDE_2017",
    )

    for variant, result in [
        (
            "ALL_YEARS",
            full_result,
        ),
        (
            "EXCLUDE_2017",
            no2017_result,
        ),
    ]:
        vegetation_rows.append(
            {
                "indicator":
                    indicator,
                "variant":
                    variant,
                "beta_c_per_1_0":
                    result["beta"],
                "response_c_per_0_10":
                    result[
                        "response_per_0_10"
                    ],
                "clustered_se":
                    result[
                        "se_clustered"
                    ],
                "p_value":
                    result[
                        "p_value"
                    ],
                "ci95_lower_per_0_10":
                    result[
                        "ci95_lower_per_0_10"
                    ],
                "ci95_upper_per_0_10":
                    result[
                        "ci95_upper_per_0_10"
                    ],
                "sign":
                    sign_label(
                        result["beta"]
                    ),
                "ci_position":
                    ci_position(
                        result[
                            "ci95_lower"
                        ],
                        result[
                            "ci95_upper"
                        ],
                    ),
                "within_r2":
                    result[
                        "within_r2"
                    ],
            }
        )

vegetation_df = pd.DataFrame(
    vegetation_rows
)

vegetation_df.to_csv(
    VEGETATION_COMPARE_PATH,
    index=False,
)

print(
    vegetation_df.to_string(
        index=False
    )
)


# =============================================================================
# 9. V3 COEFFICIENT REPRODUCTION CHECK
# =============================================================================

section(
    "9. REPRODUCE V3 FULL GREEN COEFFICIENT"
)

v3_reproduced = fit_twfe(
    panel=panel,
    years=ALL_YEARS,
    treatment=GREEN,
    controls=BASE_CONTROLS,
    model_name=
        "V3_REPRODUCTION",
)

v3_per_10pp = float(
    v3_reproduced[
        "response_per_0_10"
    ]
)

v3_reproduction_error = abs(
    v3_per_10pp
    - V3_GREEN_BETA_PER_10PP
)

print(
    f"V3 expected per +10 pp : {V3_GREEN_BETA_PER_10PP:+.12f} °C"
)

print(
    f"QA reproduced          : {v3_per_10pp:+.12f} °C"
)

print(
    f"Absolute error         : {v3_reproduction_error:.12e} °C"
)

if v3_reproduction_error > V3_BETA_REPRO_TOL:
    raise RuntimeError(
        "QA could not reproduce the V3 green coefficient."
    )


# =============================================================================
# 10. QA INTERPRETATION FLAGS
# =============================================================================

section(
    "10. QA INTERPRETATION FLAGS"
)

full_green_row = vegetation_df[
    (
        vegetation_df[
            "indicator"
        ]
        == GREEN
    )
    & (
        vegetation_df[
            "variant"
        ]
        == "ALL_YEARS"
    )
].iloc[0]

full_ndvi_row = vegetation_df[
    (
        vegetation_df[
            "indicator"
        ]
        == NDVI
    )
    & (
        vegetation_df[
            "variant"
        ]
        == "ALL_YEARS"
    )
].iloc[0]

full_evi_row = vegetation_df[
    (
        vegetation_df[
            "indicator"
        ]
        == EVI
    )
    & (
        vegetation_df[
            "variant"
        ]
        == "ALL_YEARS"
    )
].iloc[0]

green_ndvi_same_sign_response = (
    full_green_row[
        "sign"
    ]
    == full_ndvi_row[
        "sign"
    ]
)

all_three_same_sign_response = (
    len(
        {
            full_green_row["sign"],
            full_ndvi_row["sign"],
            full_evi_row["sign"],
        }
    )
    == 1
)

# Threshold provenance remains unresolved unless explicit evidence exists.
# Even when hits exist, source semantics require manual review.
if provenance_hits.empty:
    threshold_provenance_status = (
        "NOT_FOUND_IN_LOCAL_PROJECT_TEXT_FILES"
    )
else:
    threshold_provenance_status = (
        "SOURCE_HITS_FOUND_MANUAL_REVIEW_REQUIRED"
    )

qa_summary_rows = [
    {
        "qa_item":
            "Yearly green_mask-NDVI Pearson correlations all positive",
        "result":
            green_ndvi_all_positive,
        "interpretation":
            "Checks whether Otsu green percentage generally tracks continuous NDVI cross-sectionally within each year.",
    },
    {
        "qa_item":
            "Minimum yearly green_mask-NDVI Pearson correlation",
        "result":
            green_ndvi_min_corr,
        "interpretation":
            "Low/negative values would indicate poor same-year alignment in at least one year.",
    },
    {
        "qa_item":
            "Median yearly green_mask-NDVI Pearson correlation",
        "result":
            green_ndvi_median_corr,
        "interpretation":
            "Descriptive same-year alignment strength.",
    },
    {
        "qa_item":
            "Green TWFE sign reverses after adding controls",
        "result":
            green_sign_reversal_across_controls,
        "interpretation":
            "If True, positive V3 result may involve suppression/sign reversal induced by the control set.",
    },
    {
        "qa_item":
            "Full-control green_mask and NDVI response signs agree",
        "result":
            green_ndvi_same_sign_response,
        "interpretation":
            "If False, Otsu-derived green percentage and continuous vegetation index tell different longitudinal stories.",
    },
    {
        "qa_item":
            "green_mask, NDVI and EVI full-control signs all agree",
        "result":
            all_three_same_sign_response,
        "interpretation":
            "Cross-indicator consistency check only; not causal proof.",
    },
    {
        "qa_item":
            "Otsu threshold provenance status",
        "result":
            threshold_provenance_status,
        "interpretation":
            "Aggregated CSV cannot establish fixed-vs-year-specific Otsu thresholds. Source code/log must be manually verified.",
    },
]

qa_summary_df = pd.DataFrame(
    qa_summary_rows
)

qa_summary_df.to_csv(
    QA_SUMMARY_PATH,
    index=False,
)

print(
    qa_summary_df.to_string(
        index=False
    )
)


# =============================================================================
# 11. FIGURES
# =============================================================================

section(
    "11. SAVE QA FIGURES"
)

# Yearly green-NDVI correlations
fig, ax = plt.subplots(
    figsize=(10, 6)
)

ax.plot(
    yearly_df["year"],
    yearly_df[
        "pearson_green_ndvi"
    ],
    marker="o",
    label="green_mask vs NDVI",
)

ax.plot(
    yearly_df["year"],
    yearly_df[
        "pearson_green_evi"
    ],
    marker="o",
    label="green_mask vs EVI",
)

ax.axhline(
    0.0,
    linewidth=1.0,
    color="black",
)

ax.set_xlabel(
    "Year"
)

ax.set_ylabel(
    "Pearson correlation"
)

ax.set_title(
    "Yearly Alignment of Otsu Green Percentage with Vegetation Indices"
)

ax.legend()

ax.grid(
    alpha=0.2
)

plt.tight_layout()

plt.savefig(
    YEARLY_FIGURE_PATH,
    dpi=300,
    bbox_inches="tight",
)

plt.close(fig)


# Nested green coefficients
x = np.arange(
    len(
        green_nested_df
    )
)

y = green_nested_df[
    "green_beta_c_per_10pp"
].to_numpy(dtype=float)

lower = (
    y
    - green_nested_df[
        "ci95_lower_per_10pp"
    ].to_numpy(dtype=float)
)

upper = (
    green_nested_df[
        "ci95_upper_per_10pp"
    ].to_numpy(dtype=float)
    - y
)

fig, ax = plt.subplots(
    figsize=(11, 6)
)

ax.errorbar(
    x,
    y,
    yerr=np.vstack(
        [
            lower,
            upper,
        ]
    ),
    fmt="o",
    capsize=5,
)

ax.axhline(
    0.0,
    linewidth=1.0,
    color="black",
)

ax.set_xticks(
    x
)

ax.set_xticklabels(
    green_nested_df[
        "model"
    ],
    rotation=20,
    ha="right",
)

ax.set_ylabel(
    "green_mask response per +10 pp (°C)"
)

ax.set_title(
    "Nested TWFE Diagnostic for green_mask"
)

ax.grid(
    alpha=0.2
)

plt.tight_layout()

plt.savefig(
    NESTED_FIGURE_PATH,
    dpi=300,
    bbox_inches="tight",
)

plt.close(fig)


# Vegetation indicator comparison - all years only
plot_veg = vegetation_df[
    vegetation_df[
        "variant"
    ]
    == "ALL_YEARS"
].copy()

vx = np.arange(
    len(plot_veg)
)

vy = plot_veg[
    "response_c_per_0_10"
].to_numpy(dtype=float)

vlower = (
    vy
    - plot_veg[
        "ci95_lower_per_0_10"
    ].to_numpy(dtype=float)
)

vupper = (
    plot_veg[
        "ci95_upper_per_0_10"
    ].to_numpy(dtype=float)
    - vy
)

fig, ax = plt.subplots(
    figsize=(8, 6)
)

ax.errorbar(
    vx,
    vy,
    yerr=np.vstack(
        [
            vlower,
            vupper,
        ]
    ),
    fmt="o",
    capsize=5,
)

ax.axhline(
    0.0,
    linewidth=1.0,
    color="black",
)

ax.set_xticks(
    vx
)

ax.set_xticklabels(
    plot_veg[
        "indicator"
    ]
)

ax.set_ylabel(
    "Estimated LST response per +0.10 indicator units (°C)"
)

ax.set_title(
    "Full-Control TWFE Vegetation Indicator Comparison"
)

ax.grid(
    alpha=0.2
)

plt.tight_layout()

plt.savefig(
    VEGETATION_FIGURE_PATH,
    dpi=300,
    bbox_inches="tight",
)

plt.close(fig)

print(
    f"Saved: {YEARLY_FIGURE_PATH}"
)

print(
    f"Saved: {NESTED_FIGURE_PATH}"
)

print(
    f"Saved: {VEGETATION_FIGURE_PATH}"
)


# =============================================================================
# 12. STRICT AUDIT
# =============================================================================

section(
    "12. STRICT STEP-29A-QA AUDIT"
)

audit_rows = [
    {
        "check":
            "historical_base_hash",
        "status":
            "PASS",
        "details":
            EXPECTED_HASHES[
                "kaduwela_preprocessed_base.csv"
            ],
    },
    {
        "check":
            "v3_coefficient_hash",
        "status":
            "PASS",
        "details":
            EXPECTED_HASHES[
                "29a_v3_twfe_primary_coefficients.csv"
            ],
    },
    {
        "check":
            "v3s_audit_hash",
        "status":
            "PASS",
        "details":
            EXPECTED_HASHES[
                "29a_v3s_2017_sensitivity_audit.csv"
            ],
    },
    {
        "check":
            "panel_rows",
        "status":
            "PASS"
            if len(panel)
            == EXPECTED_ROWS
            else "FAIL",
        "details":
            f"actual={len(panel)}, expected={EXPECTED_ROWS}",
    },
    {
        "check":
            "panel_balanced",
        "status":
            "PASS",
        "details":
            f"{EXPECTED_GRIDS} grids x {len(ALL_YEARS)} years",
    },
    {
        "check":
            "v3_green_coefficient_reproduced",
        "status":
            "PASS"
            if v3_reproduction_error
            <= V3_BETA_REPRO_TOL
            else "FAIL",
        "details":
            f"error_per_10pp_c={v3_reproduction_error:.12e}",
    },
    {
        "check":
            "yearly_green_ndvi_rows",
        "status":
            "PASS"
            if len(yearly_df)
            == len(ALL_YEARS)
            else "FAIL",
        "details":
            f"actual={len(yearly_df)}, expected={len(ALL_YEARS)}",
    },
    {
        "check":
            "delta_variants_completed",
        "status":
            "PASS"
            if len(delta_df)
            == 2
            else "FAIL",
        "details":
            "All transitions + exclude transitions touching 2017.",
    },
    {
        "check":
            "green_nested_specs_completed",
        "status":
            "PASS"
            if len(green_nested_df)
            == len(nested_specs)
            else "FAIL",
        "details":
            f"actual={len(green_nested_df)}, expected={len(nested_specs)}",
    },
    {
        "check":
            "ndvi_nested_specs_completed",
        "status":
            "PASS"
            if len(ndvi_nested_df)
            == len(ndvi_specs)
            else "FAIL",
        "details":
            f"actual={len(ndvi_nested_df)}, expected={len(ndvi_specs)}",
    },
    {
        "check":
            "vegetation_indicator_comparison_completed",
        "status":
            "PASS"
            if len(vegetation_df)
            == 6
            else "FAIL",
        "details":
            "green_mask, NDVI, EVI x all-years/no-2017.",
    },
    {
        "check":
            "otsu_threshold_provenance_inferred_from_csv",
        "status":
            "PASS",
        "details":
            "No. Aggregated CSV is explicitly treated as insufficient to determine threshold-generation policy.",
    },
    {
        "check":
            "coefficient_sign_forced",
        "status":
            "PASS",
        "details":
            "No. All model signs are retained.",
    },
    {
        "check":
            "preferred_vegetation_indicator_auto_selected",
        "status":
            "PASS",
        "details":
            "No. QA reports evidence; it does not automatically replace green_mask with NDVI/EVI.",
    },
    {
        "check":
            "step28_or_step22b_modified",
        "status":
            "PASS",
        "details":
            "No. This QA step only reads historical/model-report inputs.",
    },
    {
        "check":
            "causal_claim",
        "status":
            "PASS",
        "details":
            "No. All outputs are observational validity diagnostics.",
    },
]

audit_df = pd.DataFrame(
    audit_rows
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

if (
    audit_df[
        "status"
    ]
    != "PASS"
).any():
    raise RuntimeError(
        "One or more Step-29A-QA audit checks failed."
    )


# =============================================================================
# 13. METHODOLOGY + REPORT
# =============================================================================

section(
    "13. SAVE QA METHODOLOGY + REPORT"
)

methodology = {
    "step":
        "29A-QA",
    "name":
        "Green-mask comparability and response validity audit",
    "historical_period":
        [
            2015,
            2025,
        ],
    "main_questions":
        [
            "Does green_mask align with NDVI/EVI within each year?",
            "Do within-grid year-to-year changes in green_mask align with NDVI changes?",
            "Does the green TWFE coefficient change sign as controls are added?",
            "Do green_mask, NDVI and EVI show consistent full-control within-grid response signs?",
            "Can local project files verify whether Otsu thresholds were fixed or year-specific?",
        ],
    "otsu_provenance_policy":
        (
            "Do not infer threshold-generation policy from the aggregated CSV. "
            "Search local text/source files; any hits require manual review."
        ),
    "nested_green_models":
        [
            {
                "name":
                    name,
                "controls":
                    controls,
            }
            for name, controls in nested_specs
        ],
    "nested_ndvi_models":
        [
            {
                "name":
                    name,
                "controls":
                    controls,
            }
            for name, controls in ndvi_specs
        ],
    "vegetation_indicators_compared":
        [
            GREEN,
            NDVI,
            EVI,
        ],
    "2017_sensitivity":
        True,
    "qa_flags":
        {
            "yearly_green_ndvi_all_positive":
                green_ndvi_all_positive,
            "minimum_yearly_green_ndvi_pearson":
                green_ndvi_min_corr,
            "median_yearly_green_ndvi_pearson":
                green_ndvi_median_corr,
            "green_sign_reversal_after_controls":
                green_sign_reversal_across_controls,
            "green_and_ndvi_full_control_same_sign":
                green_ndvi_same_sign_response,
            "green_ndvi_evi_all_same_sign":
                all_three_same_sign_response,
            "threshold_provenance_status":
                threshold_provenance_status,
        },
    "interpretation_limit":
        (
            "This step determines measurement/response consistency only. "
            "It does not establish causal cooling or select an intervention "
            "coefficient automatically."
        ),
}

METHOD_PATH.write_text(
    json.dumps(
        methodology,
        indent=2,
    ),
    encoding="utf-8",
)

report_lines = [
    "STEP 29A-QA - GREEN-MASK COMPARABILITY & RESPONSE VALIDITY AUDIT",
    "=" * 118,
    "",
    "OTSU PROVENANCE",
    "-" * 118,
    (
        f"Local project evidence status: {threshold_provenance_status}"
    ),
    (
        "The aggregated historical CSV has no Otsu-threshold-by-year field, "
        "so fixed-vs-year-specific threshold generation cannot be proven "
        "from the modeling table alone."
    ),
    "",
    "YEARLY GREEN_MASK / NDVI RELATIONSHIPS",
    "-" * 118,
    yearly_df.to_string(
        index=False
    ),
    "",
    "WITHIN-GRID DELTA RELATIONSHIPS",
    "-" * 118,
    delta_df.to_string(
        index=False
    ),
    "",
    "TWO-WAY-DEMEANED CORRELATIONS",
    "-" * 118,
    within_df.to_string(
        index=False
    ),
    "",
    "NESTED GREEN_MASK TWFE",
    "-" * 118,
    green_nested_df.to_string(
        index=False
    ),
    "",
    "NESTED NDVI TWFE",
    "-" * 118,
    ndvi_nested_df.to_string(
        index=False
    ),
    "",
    "VEGETATION INDICATOR COMPARISON",
    "-" * 118,
    vegetation_df.to_string(
        index=False
    ),
    "",
    "QA FLAGS",
    "-" * 118,
    qa_summary_df.to_string(
        index=False
    ),
    "",
    "INTERPRETATION RULE",
    "-" * 118,
    (
        "If green_mask and NDVI/EVI give conflicting longitudinal response "
        "signs, do not treat Otsu green percentage as a validated numeric "
        "cooling intervention variable without resolving the measurement "
        "definition."
    ),
    (
        "If all vegetation indicators show the same robust positive "
        "within-grid association, the dataset itself does not support a "
        "numeric vegetation-cooling response estimate under these observational "
        "specifications."
    ),
    (
        "If green-only and fully controlled green models reverse sign, "
        "interpret the positive V3 coefficient as control-sensitive/suppression-"
        "affected rather than as a simple physical green effect."
    ),
    (
        "No result is causal proof."
    ),
]

REPORT_PATH.write_text(
    "\n".join(report_lines),
    encoding="utf-8",
)

print(
    f"Saved: {METHOD_PATH}"
)

print(
    f"Saved: {REPORT_PATH}"
)


# =============================================================================
# 14. HASHES
# =============================================================================

section(
    "14. REPRODUCIBILITY HASHES"
)

hash_files = [
    BASE_PATH,
    V3_COEF_PATH,
    V3S_AUDIT_PATH,
    YEARLY_RELATION_PATH,
    DELTA_RELATION_PATH,
    WITHIN_CORRELATION_PATH,
    GREEN_NESTED_PATH,
    NDVI_NESTED_PATH,
    VEGETATION_COMPARE_PATH,
    PROVENANCE_SEARCH_PATH,
    QA_SUMMARY_PATH,
    AUDIT_PATH,
    METHOD_PATH,
    REPORT_PATH,
    YEARLY_FIGURE_PATH,
    NESTED_FIGURE_PATH,
    VEGETATION_FIGURE_PATH,
]

hash_rows = []

for path in hash_files:
    digest = sha256_file(
        path
    )

    hash_rows.append(
        {
            "file":
                str(path),
            "sha256":
                digest,
        }
    )

    print(
        f"{path.name}: {digest}"
    )

pd.DataFrame(
    hash_rows
).to_csv(
    HASH_PATH,
    index=False,
)

print(
    f"Saved: {HASH_PATH}"
)


# =============================================================================
# FINAL
# =============================================================================

section(
    "STEP 29A-QA COMPLETED SUCCESSFULLY"
)

print(
    "Green-mask comparability and response-validity audit completed.\n"
    f"Yearly green_mask-NDVI correlations all positive: "
    f"{green_ndvi_all_positive}.\n"
    f"Minimum yearly green_mask-NDVI Pearson r: "
    f"{green_ndvi_min_corr:+.6f}.\n"
    f"Median yearly green_mask-NDVI Pearson r: "
    f"{green_ndvi_median_corr:+.6f}.\n"
    f"Green TWFE sign reverses after adding controls: "
    f"{green_sign_reversal_across_controls}.\n"
    f"Full-control green_mask and NDVI response signs agree: "
    f"{green_ndvi_same_sign_response}.\n"
    f"green_mask, NDVI and EVI full-control signs all agree: "
    f"{all_three_same_sign_response}.\n"
    f"Otsu threshold provenance status: "
    f"{threshold_provenance_status}.\n"
    "The aggregated CSV alone cannot establish whether Otsu thresholds "
    "were fixed or recomputed per year.\n"
    "No coefficient sign was forced and no replacement intervention "
    "indicator was auto-selected.\n"
    "Step-28 forecasts and Step-22B model were not modified.\n"
    "\nNEXT: review these QA results before deciding the final Step-29A "
    "scenario interpretation and whether to proceed to Step 29B."
)
