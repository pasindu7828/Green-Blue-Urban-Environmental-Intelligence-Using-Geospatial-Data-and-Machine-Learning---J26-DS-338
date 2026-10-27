
"""
J26-DS-338 | Section 05 | Step 18F

EXPLAINABLE PLANNING REVIEW RULES

Inputs:
    18B - Integrated common-grid evidence
    18C - Development Conflict Index / frozen CCI
    18D - Exploratory planning typology
    18E - Historical green-loss context

Outputs:
    One review-rule CSV
    One reproducibility and QA JSON

Rules:
    - Preserve expert-AHP suitability.
    - Preserve frozen pressure and conservation indices.
    - Keep out-of-domain cells explicitly unavailable.
    - Do not combine DCI and CCI into another score.
    - Historical detection DOES NOT determine review route.
    - No official permissions, restrictions, or priorities.
"""

from pathlib import Path
from tempfile import TemporaryDirectory
import hashlib
import json

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

BASE = (
    ROOT / "data/processed/planning_conflict_intervention"
)

REPORT = (
    ROOT / "outputs/reports/planning_conflict_intervention"
)

FILES = {
    "18b": (
        BASE / "common_grid"
        / "18b_common_grid_integration_100m.csv"
    ),
    "18c": (
        BASE / "development_conflict"
        / "18c_development_conflict_index_100m.csv"
    ),
    "18d": (
        BASE / "conflict_typology"
        / "18d_planning_conflict_typology_100m.csv"
    ),
    "18e": (
        BASE / "historical_vigilance"
        / "18e_historical_green_loss_vigilance_100m.csv"
    ),
}

QA_FILES = {
    "18b": REPORT / "18b_common_grid_integration_qa.json",
    "18c": REPORT / "18c_development_conflict_index_qa.json",
    "18d": REPORT / "18d_planning_conflict_typology_qa.json",
    "18e": REPORT / "18e_historical_green_loss_vigilance_qa.json",
}

EXPECTED_STATUSES = {
    "18b": "PASS_COMMON_GRID_FINAL_EXPERT_AHP_INTEGRATION",
    "18c": "PASS_DCI_VALID_PRESSURE_DOMAIN_CCI_FROZEN_SEPARATE",
    "18d": "PASS_EXPLORATORY_PLANNING_TYPOLOGY_WITH_DOMAIN_LIMITS",
    "18e": "PASS_HISTORICAL_VIGILANCE_SUPPORT_ONLY_WITH_MISSINGNESS",
}

OUT_DIR = BASE / "planning_intervention_rules"

OUT_CSV = (
    OUT_DIR
    / "18f_explainable_planning_interventions_100m.csv"
)

OUT_QA = (
    REPORT
    / "18f_explainable_planning_interventions_qa.json"
)

N = 9240
DOMAIN_N = 4620
CP_N = 4609


# ============================================================
# 1. EXPLAINABLE REVIEW RULES
# ============================================================

# These are descriptive planning REVIEW STREAMS.
#
# They are not final priorities, land-use approvals,
# development restrictions, or legal classifications.

RULES = {

    "R00_OUTSIDE_PRESSURE_DOMAIN": {
        "review_stream": "INSUFFICIENT_PRESSURE_EVIDENCE",
        "action": (
            "Do not infer development conflict from this "
            "model. Obtain appropriate site-specific "
            "evidence before a planning decision."
        ),
        "reason": (
            "Future urbanization pressure and DCI are "
            "unavailable outside the trained model domain."
        ),
    },

    "R01_CONSERVATION_EVIDENCE_GAP": {
        "review_stream": "INCOMPLETE_CONSERVATION_EVIDENCE",
        "action": (
            "Review the missing conservation evidence. "
            "Use available pressure and suitability "
            "separately without assuming low "
            "conservation concern."
        ),
        "reason": (
            "DCI exists, but frozen Conservation Priority "
            "is unavailable for this grid."
        ),
    },

    "R02_JOINT_CONCERN": {
        "review_stream": "JOINT_PLANNING_AND_CONSERVATION_REVIEW",
        "action": (
            "Review development feasibility and "
            "green-conservation safeguards together. "
            "Check official zoning, site constraints, "
            "drainage and field conditions."
        ),
        "reason": (
            "Frozen pressure flag, lower relative "
            "suitability quartile, and upper relative "
            "Conservation Priority quartile coincide."
        ),
    },

    "R03_CONSERVATION_SAFEGUARD": {
        "review_stream": "CONSERVATION_SAFEGUARD_REVIEW",
        "action": (
            "Review mapped green assets, conservation "
            "evidence and applicable development controls "
            "before considering land-use changes."
        ),
        "reason": (
            "Pressure is flagged and frozen Conservation "
            "Priority is in its upper relative quartile."
        ),
    },

    "R04_SUITABILITY_CONSTRAINT": {
        "review_stream": "DEVELOPMENT_SUITABILITY_REVIEW",
        "action": (
            "Review the road-access, local road-density "
            "and terrain-slope limitations, alongside "
            "official zoning and other site constraints."
        ),
        "reason": (
            "Pressure is flagged while expert-AHP "
            "suitability is in its lower relative quartile."
        ),
    },

    "R05_MANAGED_GROWTH_FEASIBILITY": {
        "review_stream": "POTENTIAL_MANAGED_GROWTH_FEASIBILITY_REVIEW",
        "action": (
            "Consider a site feasibility assessment. "
            "Verify zoning, flood hazard, drainage, "
            "utilities, access and environmental "
            "restrictions before any development decision."
        ),
        "reason": (
            "Flagged pressure, upper relative suitability "
            "quartile and Conservation Priority outside "
            "its upper relative quartile."
        ),
    },

    "R06_FLAGGED_GENERAL_REVIEW": {
        "review_stream": "GENERAL_PLANNING_FEASIBILITY_REVIEW",
        "action": (
            "Review the pressure signal and local "
            "development constraints. Further assessment "
            "is required before interpreting suitability."
        ),
        "reason": (
            "Pressure is flagged, but the stricter "
            "combined review rules are not triggered."
        ),
    },

    "R07_CONSERVATION_CONTEXT": {
        "review_stream": "CONSERVATION_CONTEXT_MONITORING",
        "action": (
            "Retain conservation evidence for local "
            "planning review and monitor future changes. "
            "Do not infer an immediate development threat."
        ),
        "reason": (
            "Conservation Priority is in its upper relative "
            "quartile, but pressure is below the frozen "
            "operating threshold."
        ),
    },

    "R08_SUITABILITY_CONTEXT": {
        "review_stream": "SUITABILITY_CONTEXT_MONITORING",
        "action": (
            "Retain road-access and slope constraints "
            "for future planning assessment. No immediate "
            "pressure-triggered intervention is inferred."
        ),
        "reason": (
            "Suitability falls in the lower relative "
            "quartile, while pressure is not flagged."
        ),
    },

    "R09_GENERAL_MONITORING": {
        "review_stream": "GENERAL_EVIDENCE_MONITORING",
        "action": (
            "Retain the mapped evidence for planning "
            "review and future monitoring. No additional "
            "intervention is inferred from these rules."
        ),
        "reason": (
            "The grid does not meet the specified "
            "exploratory review combinations."
        ),
    },
}


# ============================================================
# 2. HELPERS
# ============================================================

def require(condition, label):

    ok = bool(condition)

    print(
        f"[{'PASS' if ok else 'FAIL'}] {label}"
    )

    if not ok:
        raise ValueError(f"18F QA failure: {label}")


def sha256(path):

    h = hashlib.sha256()

    with path.open("rb") as f:
        for block in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(block)

    return h.hexdigest()


def numeric(df, field):

    if field not in df.columns:
        raise KeyError(field)

    return pd.to_numeric(
        df[field],
        errors="coerce",
    ).to_numpy(dtype=float)


def validate_grid(df, label):

    require(
        len(df) == N
        and "grid_id" in df.columns
        and df.grid_id.notna().all()
        and df.grid_id.is_unique,
        f"{label}: 9,240 unique grid IDs",
    )


def align(df, ids):

    return (
        df.set_index("grid_id")
        .loc[ids]
        .reset_index()
    )


def read_optional_boolean(df, field):

    if field not in df.columns:
        raise KeyError(field)

    text = (
        df[field]
        .astype("string")
        .str.strip()
        .str.lower()
    )

    require(
        text.dropna().isin(
            ["true", "false", "1", "0"]
        ).all(),
        f"{field}: valid boolean/missing encoding",
    )

    return (
        text.isin(["true", "1"])
        .to_numpy(dtype=bool)
    )


# ============================================================
# 3. MAIN
# ============================================================

def main():

    print("=" * 78)
    print(
        "SECTION 05 | STEP 18F - "
        "EXPLAINABLE PLANNING REVIEW RULES"
    )
    print("=" * 78)

    all_inputs = (
        list(FILES.values())
        + list(QA_FILES.values())
    )

    for path in all_inputs:
        if not path.is_file():
            raise FileNotFoundError(path)

    for path in (OUT_CSV, OUT_QA):
        if path.exists():
            raise FileExistsError(
                f"Refusing to overwrite: {path}"
            )

    hashes = {
        p.relative_to(ROOT).as_posix(): sha256(p)
        for p in all_inputs
    }

    # ========================================================
    # 1. VERIFY STAGES 18B TO 18E
    # ========================================================

    print("\n[1] VERIFY COMPLETED RESEARCH STAGES")

    qa = {}

    for stage, path in QA_FILES.items():

        qa[stage] = json.loads(
            path.read_text(encoding="utf-8")
        )

        require(
            qa[stage].get("status")
            == EXPECTED_STATUSES[stage],
            f"{stage.upper()} research QA passed",
        )

    require(
        qa["18b"]["outputs"]["csv_sha256"]
        == sha256(FILES["18b"]),
        "18B dataset matches QA hash",
    )

    require(
        qa["18c"]["outputs"]["csv_sha256"]
        == sha256(FILES["18c"]),
        "18C dataset matches QA hash",
    )

    require(
        qa["18d"]["output"]["sha256"]
        == sha256(FILES["18d"]),
        "18D dataset matches QA hash",
    )

    require(
        qa["18e"]["output"]["sha256"]
        == sha256(FILES["18e"]),
        "18E dataset matches QA hash",
    )

    for stage in ("18d", "18e"):
        previous_inputs = (
            qa[stage].get("inputs_sha256", {})
            if stage == "18d"
            else qa[stage].get("input_sha256", {})
        )

        for previous_stage in (
            ("18b", "18c")
            if stage == "18d"
            else ("18b", "18d")
        ):

            key = (
                FILES[previous_stage]
                .relative_to(ROOT).as_posix()
            )

            require(
                previous_inputs.get(key)
                == sha256(FILES[previous_stage]),
                f"{stage.upper()} references "
                f"unchanged {previous_stage.upper()}",
            )

    # ========================================================
    # 2. LOAD AND ALIGN ALL EVIDENCE
    # ========================================================

    print("\n[2] LOAD AND ALIGN RESEARCH EVIDENCE")

    b = pd.read_csv(FILES["18b"])
    c = pd.read_csv(FILES["18c"])
    d = pd.read_csv(FILES["18d"])
    e = pd.read_csv(FILES["18e"])

    for name, frame in (
        ("18B", b),
        ("18C", c),
        ("18D", d),
        ("18E", e),
    ):
        validate_grid(frame, name)

    ids = b["grid_id"]

    for frame in (c, d, e):
        require(
            set(frame.grid_id) == set(ids),
            "Grid identities match Section 18B",
        )

    c = align(c, ids)
    d = align(d, ids)
    e = align(e, ids)

    domain = (
        b["pressure_domain_status_section05"]
        .eq("IN_MODEL_DOMAIN").to_numpy()
    )

    pressure = numeric(
        b, "urbanization_pressure_score"
    )

    suitability = numeric(
        b, "development_suitability_score"
    )

    cp = numeric(
        b, "primary_conservation_priority_0_1"
    )

    dci = numeric(
        c, "development_conflict_index_0_1"
    )

    cci = numeric(
        c, "conservation_conflict_index_0_1"
    )

    frozen_flag = numeric(
        d, "pressure_threshold_flag_frozen"
    )

    area = numeric(
        b, "kmc_overlap_area_m2"
    )

    common = np.isfinite(cp)
    flagged = domain & (frozen_flag == 1)

    # ========================================================
    # 3. VERIFY SOURCE SCORES AND DOMAIN RULES
    # ========================================================

    print("\n[3] VERIFY FROZEN SCORES AND DOMAINS")

    require(
        domain.sum() == DOMAIN_N
        and common.sum() == CP_N,
        "4,620 pressure cells; 4,609 CP cells",
    )

    require(
        np.array_equal(
            domain, np.isfinite(pressure)
        ),
        "Pressure domain unchanged",
    )

    require(
        np.array_equal(
            domain, np.isfinite(dci)
        ),
        "DCI domain unchanged",
    )

    require(
        np.all(~common | domain),
        "Frozen CP confined to pressure domain",
    )

    require(
        np.allclose(
            cci, cp,
            atol=1e-12,
            rtol=0,
            equal_nan=True,
        ),
        "Frozen CCI equals original CP",
    )

    require(
        np.allclose(
            dci[domain],
            pressure[domain]
            * (1 - suitability[domain]),
            atol=1e-12,
            rtol=0,
        ),
        "Development Conflict Index unchanged",
    )

    require(
        np.allclose(
            numeric(d, "development_conflict_index_0_1"),
            dci,
            atol=1e-12,
            rtol=0,
            equal_nan=True,
        ),
        "Typology uses the same DCI",
    )

    require(
        np.allclose(
            numeric(d, "conservation_conflict_index_0_1"),
            cci,
            atol=1e-12,
            rtol=0,
            equal_nan=True,
        ),
        "Typology uses the same CCI",
    )

    require(
        np.isfinite(suitability).all()
        and ((suitability >= 0)
             & (suitability <= 1)).all(),
        "Expert-AHP suitability valid on all grids",
    )

    require(
        np.isnan(frozen_flag[~domain]).all()
        and flagged.sum() == 255,
        "255 frozen pressure flags preserved",
    )

    require(
        np.isfinite(area).all()
        and np.all(area > 0)
        and np.isclose(
            area.sum() / 1e6,
            88.82921581108494,
            atol=0.001,
            rtol=0,
        ),
        "KMC clipped grid areas verified",
    )

    # ========================================================
    # 4. RECONSTRUCT EXPLORATORY TYPOLOGY FLAGS
    # ========================================================

    print("\n[4] VERIFY EXPLORATORY RULE TRIGGERS")

    low = (
        d["relative_suitability_screen"]
        .eq("LOWEST_25_PERCENT_DOMAIN_RELATIVE")
        .to_numpy()
    )

    high = (
        d["relative_suitability_screen"]
        .eq("HIGHEST_25_PERCENT_DOMAIN_RELATIVE")
        .to_numpy()
    )

    upper_cp = (
        d["relative_frozen_cp_screen"]
        .eq("HIGHEST_25_PERCENT_COMMON_DOMAIN")
        .to_numpy()
    )

    require(
        low.sum() == 1155
        and high.sum() == 1155
        and upper_cp.sum() == 1153,
        "Original relative quartile memberships preserved",
    )

    require(
        not np.any(low & high)
        and np.all(~low | domain)
        and np.all(~high | domain)
        and np.all(~upper_cp | common),
        "Quartile flags respect evidence domains",
    )

    dual = (
        flagged & low & upper_cp
    )

    managed = (
        flagged & high & common & ~upper_cp
    )

    require(
        dual.sum() == 7
        and managed.sum() == 1,
        "Original 7 dual-concern and 1 "
        "managed-growth screens reproduced",
    )

    require(
        np.array_equal(
            read_optional_boolean(
                d,
                "joint_flagged_pressure_lower_suitability_upper_cp",
            )[common],
            dual[common],
        ),
        "Stored joint-concern flags reproduced",
    )

    require(
        np.array_equal(
            read_optional_boolean(
                d,
                "preliminary_managed_growth_screening_candidate",
            )[common],
            managed[common],
        ),
        "Stored growth-screen flags reproduced",
    )

    # ========================================================
    # 5. ASSIGN NONOVERLAPPING REVIEW RULES
    # ========================================================

    print("\n[5] APPLY EXPLAINABLE REVIEW RULES")

    routes = np.full(
        N, "UNASSIGNED", dtype=object
    )

    route_conditions = [

        (
            "R00_OUTSIDE_PRESSURE_DOMAIN",
            ~domain,
        ),

        (
            "R01_CONSERVATION_EVIDENCE_GAP",
            domain & ~common,
        ),

        (
            "R02_JOINT_CONCERN",
            common & dual,
        ),

        (
            "R03_CONSERVATION_SAFEGUARD",
            common & flagged & upper_cp & ~dual,
        ),

        (
            "R04_SUITABILITY_CONSTRAINT",
            common & flagged & low & ~upper_cp,
        ),

        (
            "R05_MANAGED_GROWTH_FEASIBILITY",
            common & managed,
        ),

        (
            "R06_FLAGGED_GENERAL_REVIEW",
            common & flagged
            & ~upper_cp & ~low & ~managed,
        ),

        (
            "R07_CONSERVATION_CONTEXT",
            common & ~flagged & upper_cp,
        ),

        (
            "R08_SUITABILITY_CONTEXT",
            common & ~flagged & low & ~upper_cp,
        ),

        (
            "R09_GENERAL_MONITORING",
            common & ~flagged & ~upper_cp & ~low,
        ),

    ]

    for rule_id, mask in route_conditions:

        require(
            not np.any(
                mask & (routes != "UNASSIGNED")
            ),
            f"No overlapping assignments for {rule_id}",
        )

        routes[mask] = rule_id

    require(
        np.all(routes != "UNASSIGNED"),
        "All grids receive exactly one review rule",
    )

    require(
        np.all(
            routes[~domain]
            == "R00_OUTSIDE_PRESSURE_DOMAIN"
        ),
        "Out-of-domain grids make no conflict inference",
    )

    require(
        np.all(
            routes[domain & ~common]
            == "R01_CONSERVATION_EVIDENCE_GAP"
        ),
        "11 conservation evidence gaps protected",
    )

    require(
        int(np.sum(
            routes == "R02_JOINT_CONCERN"
        )) == 7,
        "Seven joint-concern review streams preserved",
    )

    require(
        int(np.sum(
            routes == "R05_MANAGED_GROWTH_FEASIBILITY"
        )) == 1,
        "One feasibility screening stream preserved",
    )

    print("\nReview-stream distribution:")

    counts = pd.Series(routes).value_counts()

    for rule_id, count in counts.items():
        print(f"{rule_id}: {count}")

    # ========================================================
    # 6. ATTACH HISTORICAL CONTEXT ONLY
    # ========================================================

    print("\n[6] ATTACH HISTORICAL SUPPORTING EVIDENCE")

    historical_fields = [
        "historical_evidence_context_type",
        "historical_g2b_robust_exposure_share_250m",
        "historical_g2b_robust_exposure_share_500m",
        "historical_g2b_strict_exposure_share_250m",
        "historical_g2b_strict_exposure_share_500m",
    ]

    require(
        set(historical_fields).issubset(e.columns),
        "Historical evidence fields available",
    )

    require(
        e["historical_evidence_context_type"]
        .notna().all(),
        "Historical context categories complete",
    )

    require(
        np.allclose(
            numeric(
                e,
                "historical_g2b_robust_exposure_share_250m",
            ),
            numeric(
                b,
                "historical_g2b_robust_exposure_share_250m",
            ),
            atol=1e-12,
            rtol=0,
            equal_nan=True,
        ),
        "Historical exposure remains frozen",
    )

    # The previous step found robust detection in
    # 9,239/9,240 grids at 500m.
    #
    # This is nearly ubiquitous. Therefore:
    #
    # - Historical binary detection does not select routes.
    # - Historical exposure measures stay explanatory.
    # - No extra index weights are assigned.
    # - No historical conversion is called illegal.

    # ========================================================
    # 7. CREATE EXPLAINABLE OUTPUT DATASET
    # ========================================================

    print("\n[7] BUILD REVIEW AND EXPLANATION TABLE")

    review_stream = [
        RULES[r]["review_stream"]
        for r in routes
    ]

    review_action = [
        RULES[r]["action"]
        for r in routes
    ]

    rule_reason = [
        RULES[r]["reason"]
        for r in routes
    ]

    result = pd.DataFrame({

        "grid_id": b["grid_id"],

        "kmc_overlap_area_m2": area,

        "pressure_origin_year": 2025,

        "pressure_period": "2026-2028",

        "pressure_domain_status":
            b["pressure_domain_status_section05"],

        "urbanization_pressure_score":
            pressure,

        "pressure_threshold_flag_frozen":
            pd.Series(frozen_flag).astype("Int64"),

        "development_suitability_score":
            suitability,

        "development_conflict_index_0_1":
            dci,

        "frozen_conservation_conflict_index_0_1":
            cci,

        "relative_suitability_screen":
            d["relative_suitability_screen"],

        "relative_frozen_cp_screen":
            d["relative_frozen_cp_screen"],

        "planning_evidence_type":
            d["planning_evidence_type"],

        "review_rule_id": routes,

        "review_stream": review_stream,

        "review_action": review_action,

        "explanation_of_rule_trigger": rule_reason,

        "historical_context_type":
            e["historical_evidence_context_type"],

        "historical_robust_exposure_share_250m":
            numeric(
                e,
                "historical_g2b_robust_exposure_share_250m",
            ),

        "historical_robust_exposure_share_500m":
            numeric(
                e,
                "historical_g2b_robust_exposure_share_500m",
            ),

        "historical_strict_exposure_share_250m":
            numeric(
                e,
                "historical_g2b_strict_exposure_share_250m",
            ),

        "historical_strict_exposure_share_500m":
            numeric(
                e,
                "historical_g2b_strict_exposure_share_500m",
            ),

        "historical_context_used_as_score_weight":
            False,

        "legal_or_policy_decision_assigned":
            False,

        "intervention_priority_rank_assigned":
            False,

        "field_and_official_planning_verification_required":
            True,

        "methodology":
            "TRANSPARENT_EXPLORATORY_RULES_NOT_POLICY",
    })

    validate_grid(
        result,
        "18F explainable review table",
    )

    require(
        np.allclose(
            numeric(
                result,
                "development_conflict_index_0_1",
            ),
            dci,
            atol=1e-12,
            rtol=0,
            equal_nan=True,
        ),
        "Output DCI unchanged",
    )

    require(
        np.allclose(
            numeric(
                result,
                "frozen_conservation_conflict_index_0_1",
            ),
            cci,
            atol=1e-12,
            rtol=0,
            equal_nan=True,
        ),
        "Output frozen CCI unchanged",
    )

    # ========================================================
    # 8. EXPORT WITHOUT OVERWRITING
    # ========================================================

    print("\n[8] EXPORT AND VALIDATE")

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with TemporaryDirectory(
        prefix="18f_build_",
        dir=OUT_DIR,
    ) as folder:

        temporary = (
            Path(folder) / OUT_CSV.name
        )

        result.to_csv(
            temporary,
            index=False,
        )

        saved = pd.read_csv(temporary)

        validate_grid(
            saved,
            "Exported review table",
        )

        require(
            saved["review_rule_id"]
            .equals(result["review_rule_id"]),
            "Rule identifiers survive export",
        )

        require(
            saved["review_action"]
            .equals(result["review_action"]),
            "Review actions survive export",
        )

        for col in (
            "development_conflict_index_0_1",
            "frozen_conservation_conflict_index_0_1",
        ):

            require(
                np.allclose(
                    numeric(saved, col),
                    numeric(result, col),
                    equal_nan=True,
                    atol=1e-12,
                    rtol=0,
                ),
                f"Exported {col} unchanged",
            )

        require(
            not OUT_CSV.exists()
            and not OUT_QA.exists(),
            "No existing outputs overwritten",
        )

        temporary.replace(OUT_CSV)

    # ========================================================
    # 9. VERIFY SOURCE FILE INTEGRITY
    # ========================================================

    print("\n[9] VERIFY FROZEN INPUT INTEGRITY")

    for path in all_inputs:

        key = path.relative_to(ROOT).as_posix()

        require(
            sha256(path) == hashes[key],
            f"Unmodified: {path.name}",
        )

    # ========================================================
    # 10. WRITE QA REPORT
    # ========================================================

    print("\n[10] WRITE REPRODUCIBILITY REPORT")

    report = {

        "section": "05_planning_conflict_intervention",

        "stage": "18f_explainable_planning_interventions",

        "status": (
            "PASS_EXPLAINABLE_PLANNING_REVIEW_RULES_"
            "PRIORITIZATION_PENDING"
        ),

        "grid_cells": N,

        "pressure_domain_cells": int(domain.sum()),

        "conservation_common_domain_cells":
            int(common.sum()),

        "out_of_domain_cells": int((~domain).sum()),

        "missing_conservation_cells_inside_pressure_domain":
            int((domain & ~common).sum()),

        "frozen_pressure_flagged_cells":
            int(flagged.sum()),

        "review_rule_counts": {
            rule_id: int(np.sum(routes == rule_id))
            for rule_id in RULES
        },

        "rulebook": RULES,

        "screening_basis": [
            "Frozen pressure threshold 0.715",
            "Relative suitability lower/upper quartiles",
            "Frozen CP upper relative quartile",
            "Strict preservation of model-domain missingness",
        ],

        "frozen_cp_contains_pressure_information": True,

        "dci_and_cci_combined_into_new_risk_score": False,

        "historical_conversion_used_for_rule_routing": False,

        "historical_binary_detection_discriminative": False,

        "historical_context_role":
            "EXPLANATION_AND_SITE_VERIFICATION_ONLY",

        "official_planning_decisions_issued": False,

        "intervention_priority_ranks_issued": False,

        "expert_ahp_weights_modified": False,

        "sections02_03_04_modified": False,

        "limitations": [
            (
                "All intervention outputs are recommendations "
                "for further planning review, not approvals "
                "or mandatory restrictions."
            ),
            (
                "Exploratory quartile boundaries are not "
                "official policy thresholds."
            ),
            (
                "Conservation Priority contains urbanization "
                "pressure information and is therefore not "
                "an independent pressure predictor."
            ),
            (
                "Future pressure is a model-derived "
                "relative score, not a calibrated probability."
            ),
            (
                "Historical conversion occurs around almost "
                "all cells at 500m and is not useful as "
                "a simple binary prioritization signal."
            ),
            (
                "No official zoning, drainage, flood hazard "
                "or legal compliance determination is made."
            ),
            (
                "Any subsequent prioritization must keep "
                "DCI and CCI conceptually separate and "
                "must test sensitivity to screening cutoffs."
            ),
        ],

        "input_sha256": hashes,

        "output": {
            "csv":
                OUT_CSV.relative_to(ROOT).as_posix(),
            "sha256": sha256(OUT_CSV),
        },
    }

    OUT_QA.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print("\n" + "=" * 78)

    print(report["status"])

    print("Grid cells:", N)
    print("Common evidence:", int(common.sum()))
    print("Out of pressure domain:", int((~domain).sum()))

    print(
        "Joint-concern review streams:",
        int(np.sum(
            routes == "R02_JOINT_CONCERN"
        )),
    )

    print(
        "Managed-growth feasibility review streams:",
        int(np.sum(
            routes == "R05_MANAGED_GROWTH_FEASIBILITY"
        )),
    )

    print("Historical evidence used for score weighting: NO")
    print("Expert weights modified: NO")
    print("Official decisions issued: NO")
    print("Final priority ranking assigned: NO")

    print("\nCSV:", OUT_CSV)
    print("QA:", OUT_QA)

    print("=" * 78)


if __name__ == "__main__":
    main()
