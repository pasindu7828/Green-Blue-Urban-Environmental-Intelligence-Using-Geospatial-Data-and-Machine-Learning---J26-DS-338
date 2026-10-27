
"""
J26-DS-338 | Section 05 | Step 18D

EXPLORATORY PLANNING CONFLICT TYPOLOGY

Inputs:
    18B - Common-grid integrated evidence
    18C - Development Conflict Index and frozen CCI

Three independent dimensions:
    1. Frozen urbanization-pressure threshold
    2. Development Suitability relative quartiles
    3. Frozen Conservation Priority relative quartile

Relative quartiles are ANALYTICAL SCREENING RULES.
They are not expert-derived policy thresholds or
official land-development classifications.

No predictions, retraining, or modifications to
Sections 02, 03 or 04.

Historical vigilance and intervention priorities
will be handled in subsequent research steps.
"""

from pathlib import Path
from tempfile import TemporaryDirectory
import hashlib
import json
import math

import numpy as np
import pandas as pd


# ============================================================
# 1. PROJECT PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

BASE = (
    ROOT / "data/processed/planning_conflict_intervention"
)

REPORT = (
    ROOT / "outputs/reports/planning_conflict_intervention"
)

INPUT_GRID = (
    BASE / "common_grid"
    / "18b_common_grid_integration_100m.csv"
)

INPUT_GRID_QA = (
    REPORT / "18b_common_grid_integration_qa.json"
)

INPUT_DCI = (
    BASE / "development_conflict"
    / "18c_development_conflict_index_100m.csv"
)

INPUT_DCI_QA = (
    REPORT / "18c_development_conflict_index_qa.json"
)

OUT_DIR = BASE / "conflict_typology"

OUT_CSV = (
    OUT_DIR / "18d_planning_conflict_typology_100m.csv"
)

OUT_QA = (
    REPORT / "18d_planning_conflict_typology_qa.json"
)


# ============================================================
# 2. VERIFIED RESEARCH CONTRACT
# ============================================================

N = 9240

DOMAIN_N = 4620

CP_N = 4609

FLAGGED_N = 255

THRESHOLD = 0.715


# ============================================================
# 3. HELPERS
# ============================================================

def require(ok, text):

    if not bool(ok):
        raise ValueError(
            f"[FAIL] {text}"
        )

    print(f"[PASS] {text}")


def sha256(path):

    h = hashlib.sha256()

    with path.open("rb") as f:

        for block in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(block)

    return h.hexdigest()


def numbers(df, name):

    require(
        name in df.columns,
        f"Field exists: {name}",
    )

    return pd.to_numeric(
        df[name],
        errors="coerce",
    ).to_numpy(dtype=float)


def top_n_mask(
    scores,
    grid_ids,
    valid,
    n,
    largest=True,
):

    """
    Select exactly n cells by score.

    Tie-breaking:
        Ascending grid_id.

    This is deterministic, reproducible,
    and does not modify source scores.
    """

    idx = np.flatnonzero(valid)

    require(
        len(idx) >= n > 0,
        f"Valid ranking domain for top {n}",
    )

    values = scores[idx]

    require(
        np.isfinite(values).all(),
        "Ranked values are finite",
    )

    ids = grid_ids[idx]

    keys = -values if largest else values

    order = np.lexsort(
        (ids, keys)
    )

    output = np.zeros(
        len(scores),
        dtype=bool,
    )

    output[idx[order[:n]]] = True

    require(
        output.sum() == n,
        f"Exactly {n} ranked cells selected",
    )

    return output


# ============================================================
# 4. MAIN WORKFLOW
# ============================================================

def main():

    print("=" * 78)
    print(
        "SECTION 05 | STEP 18D - "
        "PLANNING CONFLICT EVIDENCE TYPOLOGY"
    )
    print("=" * 78)

    inputs = (
        INPUT_GRID,
        INPUT_GRID_QA,
        INPUT_DCI,
        INPUT_DCI_QA,
    )

    for path in inputs:

        if not path.is_file():
            raise FileNotFoundError(path)

    for path in (
        OUT_CSV,
        OUT_QA,
    ):

        if path.exists():
            raise FileExistsError(
                f"Output exists; not overwriting: {path}"
            )

    hashes = {
        p.relative_to(ROOT).as_posix(): sha256(p)
        for p in inputs
    }

    # ========================================================
    # STEP 1 - VERIFY SOURCE RELEASES
    # ========================================================

    print("\n[1] VERIFY SOURCE RELEASES")

    grid_qa = json.loads(
        INPUT_GRID_QA.read_text(
            encoding="utf-8"
        )
    )

    dci_qa = json.loads(
        INPUT_DCI_QA.read_text(
            encoding="utf-8"
        )
    )

    require(
        grid_qa.get("status")
        == "PASS_COMMON_GRID_FINAL_EXPERT_AHP_INTEGRATION",
        "18B QA passed",
    )

    require(
        dci_qa.get("status")
        == "PASS_DCI_VALID_PRESSURE_DOMAIN_CCI_FROZEN_SEPARATE",
        "18C QA passed",
    )

    require(
        grid_qa["outputs"]["csv_sha256"]
        == sha256(INPUT_GRID),
        "18B CSV hash unchanged",
    )

    require(
        dci_qa["outputs"]["csv_sha256"]
        == sha256(INPUT_DCI),
        "18C CSV hash unchanged",
    )

    require(
        dci_qa["input_sha256"][
            INPUT_GRID.relative_to(ROOT).as_posix()
        ] == sha256(INPUT_GRID),
        "18C derived from exact 18B data",
    )

    # ========================================================
    # STEP 2 - LOAD AND RECONCILE GRID DATA
    # ========================================================

    print("\n[2] READ AND RECONCILE ROWS")

    grid = pd.read_csv(INPUT_GRID)
    conflicts = pd.read_csv(INPUT_DCI)

    for name, df in [
        ("18B", grid),
        ("18C", conflicts),
    ]:

        require(
            len(df) == N
            and df["grid_id"].notna().all()
            and df["grid_id"].is_unique,
            f"{name}: 9,240 unique cells",
        )

    require(
        set(grid.grid_id)
        == set(conflicts.grid_id),
        "Same grid IDs across 18B and 18C",
    )

    conflicts = (
        conflicts.set_index("grid_id")
        .loc[grid.grid_id]
        .reset_index()
    )

    grid_id = (
        grid.grid_id
        .astype(str)
        .to_numpy()
    )

    pressure = numbers(
        grid,
        "urbanization_pressure_score",
    )

    suitability = numbers(
        grid,
        "development_suitability_score",
    )

    cp = numbers(
        grid,
        "primary_conservation_priority_0_1",
    )

    frozen_flag = numbers(
        grid,
        "threshold_flag",
    )

    dci = numbers(
        conflicts,
        "development_conflict_index_0_1",
    )

    cci = numbers(
        conflicts,
        "conservation_conflict_index_0_1",
    )

    cp_shortfall = numbers(
        conflicts,
        "suitability_shortfall_0_1",
    )

    source_p = numbers(
        conflicts,
        "urbanization_pressure_score",
    )

    source_s = numbers(
        conflicts,
        "development_suitability_score",
    )

    area = numbers(
        grid,
        "kmc_overlap_area_m2",
    )

    domain = (
        grid["pressure_domain_status_section05"]
        .eq("IN_MODEL_DOMAIN")
        .to_numpy()
    )

    cci_valid = np.isfinite(cci)

    flagged = (
        domain & (frozen_flag == 1)
    )

    # ========================================================
    # STEP 3 - VERIFY THE FROZEN INPUTS
    # ========================================================

    require(
        domain.sum() == DOMAIN_N,
        "4,620 valid pressure-domain cells",
    )

    require(
        np.array_equal(
            domain,
            np.isfinite(pressure),
        ),
        "Pressure missingness matches domain",
    )

    require(
        np.isfinite(suitability).all()
        and (
            (suitability >= 0)
            & (suitability <= 1)
        ).all(),
        "All suitability scores finite and bounded",
    )

    require(
        np.array_equal(
            cci_valid,
            np.isfinite(cp),
        )
        and cci_valid.sum() == CP_N,
        "CCI availability matches 4,609 frozen CP values",
    )

    require(
        (~domain & cci_valid).sum() == 0,
        "CCI absent outside pressure domain",
    )

    require(
        np.array_equal(
            domain,
            np.isfinite(dci),
        ),
        "DCI matches pressure domain",
    )

    require(
        np.isnan(
            frozen_flag[~domain]
        ).all()
        and flagged.sum() == FLAGGED_N,
        "255 original pressure threshold flags preserved",
    )

    require(
        np.allclose(
            pressure,
            source_p,
            atol=1e-12,
            rtol=0,
            equal_nan=True,
        ),
        "18C pressure preserved",
    )

    require(
        np.allclose(
            suitability,
            source_s,
            atol=1e-12,
            rtol=0,
        ),
        "18C suitability preserved",
    )

    require(
        np.allclose(
            cci,
            cp,
            atol=1e-12,
            rtol=0,
            equal_nan=True,
        ),
        "18C CCI equals frozen CP",
    )

    require(
        np.allclose(
            cp_shortfall,
            1.0 - suitability,
            atol=1e-12,
            rtol=0,
        ),
        "18C shortfall preserved",
    )

    require(
        np.allclose(
            dci[domain],
            pressure[domain]
            * (1.0 - suitability[domain]),
            atol=1e-12,
            rtol=0,
        ),
        "DCI formula reproduced",
    )

    require(
        np.all(np.isfinite(area))
        and np.all(area > 0)
        and np.isclose(
            area.sum() / 1e6,
            88.82921581108494,
            atol=0.001,
            rtol=0,
        ),
        "Valid clipped KMC grid areas",
    )

    require(
        np.array_equal(
            grid["dci_input_status"]
            .eq("READY_FOR_18C_DCI")
            .to_numpy(),
            domain,
        ),
        "18B DCI readiness matches domain",
    )

    require(
        np.array_equal(
            conflicts["development_conflict_status"]
            .eq("VALID_PRESSURE_SUITABILITY_DOMAIN")
            .to_numpy(),
            domain,
        ),
        "18C DCI status matches domain",
    )

    # ========================================================
    # STEP 4 - RELATIVE SCREENING
    # ========================================================

    print(
        "\n[3] COMPUTE DESCRIPTIVE "
        "DOMAIN-RELATIVE SCREENING FLAGS"
    )

    # Important methodological distinction:
    #
    # These quartiles describe the distribution
    # within each available research domain.
    #
    # They are NOT legal thresholds or additional
    # expert-assigned criterion weights.
    #
    # Suitability:
    #   Lowest 25% of pressure-domain grids
    #   Middle 50%
    #   Highest 25%
    #
    # Conservation:
    #   Highest 25% of frozen CP common-domain grids
    #   Remaining 75%
    #
    # Exact membership is deterministic by grid_id.

    quarter_s = int(
        math.ceil(DOMAIN_N * 0.25)
    )

    quarter_cp = int(
        math.ceil(CP_N * 0.25)
    )

    low_s = top_n_mask(
        suitability,
        grid_id,
        domain,
        quarter_s,
        largest=False,
    )

    high_s = top_n_mask(
        suitability,
        grid_id,
        domain,
        quarter_s,
        largest=True,
    )

    upper_cp = top_n_mask(
        cci,
        grid_id,
        cci_valid,
        quarter_cp,
        largest=True,
    )

    require(
        not np.any(low_s & high_s),
        "Suitability outer quartiles do not overlap",
    )

    require(
        np.all(~low_s | domain)
        and np.all(~high_s | domain),
        "Suitability screening inside pressure domain",
    )

    require(
        np.all(~upper_cp | cci_valid),
        "Conservation quartile restricted to CP domain",
    )

    # ========================================================
    # STEP 5 - ASSIGN INDEPENDENT SIGNALS
    # ========================================================

    suit_tier = np.full(
        N,
        "NOT_ASSESSED_OUTSIDE_PRESSURE_DOMAIN",
        dtype=object,
    )

    suit_tier[domain] = (
        "MIDDLE_50_PERCENT_DOMAIN_RELATIVE"
    )

    suit_tier[low_s] = (
        "LOWEST_25_PERCENT_DOMAIN_RELATIVE"
    )

    suit_tier[high_s] = (
        "HIGHEST_25_PERCENT_DOMAIN_RELATIVE"
    )

    cp_tier = np.full(
        N,
        "FROZEN_CP_UNAVAILABLE",
        dtype=object,
    )

    cp_tier[cci_valid] = (
        "OTHER_75_PERCENT_COMMON_DOMAIN"
    )

    cp_tier[upper_cp] = (
        "HIGHEST_25_PERCENT_COMMON_DOMAIN"
    )

    pressure_tier = np.full(
        N,
        "NOT_IN_PRESSURE_MODEL_DOMAIN",
        dtype=object,
    )

    pressure_tier[domain] = (
        "BELOW_FROZEN_PRESSURE_THRESHOLD"
    )

    pressure_tier[flagged] = (
        "FLAGGED_AT_FROZEN_THRESHOLD"
    )

    # ========================================================
    # STEP 6 - CONSTRUCT COMBINED EVIDENCE TYPES
    # ========================================================

    label_s = np.full(
        N,
        "S_MIDDLE50",
        dtype=object,
    )

    label_s[low_s] = "S_LOWER25"
    label_s[high_s] = "S_UPPER25"

    label_p = np.where(
        flagged,
        "P_FLAGGED",
        "P_UNFLAGGED",
    )

    label_cp = np.where(
        upper_cp,
        "CP_UPPER25",
        "CP_OTHER75",
    )

    planning_type = np.full(
        N,
        "OUTSIDE_PRESSURE_MODEL_DOMAIN",
        dtype=object,
    )

    common = domain & cci_valid

    planning_type[common] = np.char.add(
        np.char.add(
            np.char.add(
                label_p[common].astype(str),
                "__",
            ),
            label_s[common].astype(str),
        ),
        np.char.add(
            "__",
            label_cp[common].astype(str),
        ),
    )

    planning_type[
        domain & ~cci_valid
    ] = (
        "DCI_AVAILABLE__FROZEN_CP_MISSING"
    )

    # Independent multi-signal screening flags.
    #
    # These are preliminary decision-support signals,
    # not final intervention instructions.

    managed_growth_screen = (
        flagged
        & high_s
        & cci_valid
        & ~upper_cp
    )

    dual_concern_screen = (
        flagged
        & low_s
        & upper_cp
    )

    flagged_conservation_screen = (
        flagged & upper_cp
    )

    flagged_suitability_screen = (
        flagged & low_s
    )

    require(
        (
            planning_type
            == "OUTSIDE_PRESSURE_MODEL_DOMAIN"
        ).sum() == N - DOMAIN_N,
        "No typology assertions outside domain",
    )

    require(
        (
            planning_type
            == "DCI_AVAILABLE__FROZEN_CP_MISSING"
        ).sum() == 11,
        "11 partial-evidence grids identified",
    )

    require(
        int(common.sum()) == CP_N,
        "Full three-signal typology assigned "
        "to 4,609 common-domain cells",
    )

    require(
        not np.any(
            managed_growth_screen
            & dual_concern_screen
        ),
        "Growth candidates and dual-concern "
        "flags mutually exclusive",
    )

    require(
        np.all(managed_growth_screen <= common),
        "Growth screening only with "
        "all required evidence",
    )

    # ========================================================
    # STEP 7 - CREATE TYPOLOGY TABLE
    # ========================================================

    print(
        "\n[4] BUILD AUDITABLE TYPOLOGY OUTPUT"
    )

    result = pd.DataFrame({

        "grid_id":
            grid.grid_id,

        "kmc_overlap_area_m2":
            area,

        "pressure_origin_year":
            2025,

        "pressure_period":
            "2026-2028",

        "urbanization_pressure_score":
            pressure,

        "pressure_threshold_flag_frozen":
            pd.Series(frozen_flag).astype("Int64"),

        "pressure_signal":
            pressure_tier,

        "development_suitability_score":
            suitability,

        "development_conflict_index_0_1":
            dci,

        "relative_suitability_screen":
            suit_tier,

        "conservation_conflict_index_0_1":
            cci,

        "relative_frozen_cp_screen":
            cp_tier,

        "planning_evidence_type":
            planning_type,

        "flagged_pressure_and_lower_suitability_quartile":
            pd.Series(
                flagged_suitability_screen
            ).where(domain).astype("boolean"),

        "flagged_pressure_and_upper_cp_quartile":
            pd.Series(
                flagged_conservation_screen
            ).where(cci_valid).astype("boolean"),

        "joint_flagged_pressure_lower_suitability_upper_cp":
            pd.Series(
                dual_concern_screen
            ).where(cci_valid).astype("boolean"),

        "preliminary_managed_growth_screening_candidate":
            pd.Series(
                managed_growth_screen
            ).where(cci_valid).astype("boolean"),

        "typing_basis":
            "FROZEN_PRESSURE_THRESHOLD_AND_RELATIVE_QUARTILES",

        "is_official_land_use_classification":
            False,

    })

    require(
        len(result) == N
        and result.grid_id.is_unique,
        "Final typology 9,240 unique rows",
    )

    require(
        np.array_equal(
            result.planning_evidence_type.eq(
                "OUTSIDE_PRESSURE_MODEL_DOMAIN"
            ).to_numpy(),
            ~domain,
        ),
        "Explicit out-of-domain typology",
    )

    print("\nEvidence-type frequencies:")

    print(
        result.planning_evidence_type
        .value_counts()
        .to_string()
    )

    # ========================================================
    # STEP 8 - EXPORT WITH VERIFICATION
    # ========================================================

    print("\n[5] EXPORT AND ROUNDTRIP CHECK")

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORT.mkdir(
        parents=True,
        exist_ok=True,
    )

    with TemporaryDirectory(
        prefix="18d_build_",
        dir=OUT_DIR,
    ) as tmp:

        tmp_csv = Path(tmp) / OUT_CSV.name

        result.to_csv(
            tmp_csv,
            index=False,
        )

        loaded = pd.read_csv(
            tmp_csv
        )

        require(
            len(loaded) == N
            and loaded.grid_id.is_unique,
            "Exported rows preserved",
        )

        require(
            loaded.planning_evidence_type
            .fillna("")
            .equals(
                result.planning_evidence_type.fillna("")
            ),
            "Evidence type labels preserved",
        )

        for field, expected in [

            (
                "development_conflict_index_0_1",
                dci,
            ),

            (
                "conservation_conflict_index_0_1",
                cci,
            ),

            (
                "urbanization_pressure_score",
                pressure,
            ),

        ]:

            observed = numbers(
                loaded,
                field,
            )

            require(
                np.allclose(
                    observed,
                    expected,
                    atol=1e-12,
                    rtol=0,
                    equal_nan=True,
                ),
                f"Export preserves {field} and missingness",
            )

        require(
            not OUT_CSV.exists()
            and not OUT_QA.exists(),
            "No existing output overwritten",
        )

        tmp_csv.replace(
            OUT_CSV
        )

    # ========================================================
    # STEP 9 - VERIFY INPUT INTEGRITY
    # ========================================================

    print("\n[6] VERIFY INPUT INTEGRITY")

    for p in inputs:

        require(
            sha256(p)
            == hashes[
                p.relative_to(ROOT).as_posix()
            ],
            f"Source unchanged: {p.name}",
        )

    # ========================================================
    # STEP 10 - FINAL QA REPORT
    # ========================================================

    qa = {

        "section":
            "05_planning_conflict_intervention",

        "stage":
            "18d_planning_conflict_typology",

        "status": (
            "PASS_EXPLORATORY_PLANNING_TYPOLOGY_"
            "WITH_DOMAIN_LIMITS"
        ),

        "total_grids":
            N,

        "pressure_domain_grids":
            DOMAIN_N,

        "outside_pressure_domain":
            N - DOMAIN_N,

        "common_pressure_suitability_conservation_domain":
            CP_N,

        "pressure_flagged_count":
            int(flagged.sum()),

        "frozen_pressure_threshold":
            THRESHOLD,

        "frozen_pressure_is_calibrated_probability":
            False,

        "relative_quartile_definition": (
            "Exactly ceil(25% of valid domain); "
            "ties resolved by ascending grid_id. "
            "Suitability quartiles computed within "
            "the 4,620-cell pressure domain; "
            "CP upper quartile within 4,609 "
            "available frozen CP cells."
        ),

        "suitability_lower_quartile_count":
            int(low_s.sum()),

        "suitability_upper_quartile_count":
            int(high_s.sum()),

        "cp_upper_quartile_count":
            int(upper_cp.sum()),

        "suitability_lower_quartile_max_score":
            float(suitability[low_s].max()),

        "suitability_upper_quartile_min_score":
            float(suitability[high_s].min()),

        "cp_upper_quartile_min_score":
            float(cci[upper_cp].min()),

        "counts_by_type": {
            str(k): int(v)
            for k, v in result.planning_evidence_type
            .value_counts().items()
        },

        "joint_flagged_pressure_lower_suitability_upper_cp":
            int(dual_concern_screen.sum()),

        "preliminary_managed_growth_screening_candidates":
            int(managed_growth_screen.sum()),

        "limitations": [

            (
                "Relative quartiles are descriptive "
                "analytical screening choices, not "
                "expert-derived policy cutoffs or "
                "legal planning classifications."
            ),

            (
                "Managed-growth screening candidates "
                "require zoning, flood-risk, "
                "infrastructure and ground validation."
            ),

            (
                "Pressure scores are model-derived "
                "relative scores, not calibrated "
                "probabilities."
            ),

            (
                "DCI and frozen CP are separate "
                "non-equivalent indices; their "
                "magnitudes are not interchangeable."
            ),

            (
                "CP is not recomputed or multiplied "
                "by pressure again."
            ),

            (
                "Out-of-domain pressure and DCI "
                "remain unavailable; no imputation "
                "with zero."
            ),

            (
                "The reported engineer AHP consultation "
                "informs suitability, not exploratory "
                "quartile thresholds."
            ),

            (
                "Boundary sensitivity will be "
                "evaluated in a later Section 05 step."
            ),

        ],

        "classification_not_official":
            True,

        "expert_weights_unchanged":
            True,

        "section04_modified":
            False,

        "inputs_sha256":
            hashes,

        "output": {

            "csv":
                OUT_CSV.relative_to(ROOT).as_posix(),

            "sha256":
                sha256(OUT_CSV),

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
    # STEP 11 - FINAL TERMINAL SUMMARY
    # ========================================================

    print("\n" + "=" * 78)

    print(qa["status"])

    print(
        "Common-domain descriptive types:",
        CP_N,
    )

    print(
        "Pressure-only, CP unavailable:",
        11,
    )

    print(
        "Outside pressure domain:",
        N - DOMAIN_N,
    )

    print(
        "Joint flagged pressure + "
        "lower suitability + upper CP:",
        int(dual_concern_screen.sum()),
    )

    print(
        "Preliminary managed-growth "
        "screening candidates:",
        int(managed_growth_screen.sum()),
    )

    print(
        "Official planning classifications assigned: NO"
    )

    print(
        "Intervention priorities assigned: NO"
    )

    print(
        "Source files modified: NO"
    )

    print("\nCSV:", OUT_CSV)
    print("QA:", OUT_QA)

    print("=" * 78)


if __name__ == "__main__":
    main()
