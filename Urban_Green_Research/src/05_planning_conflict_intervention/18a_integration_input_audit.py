
"""
J26-DS-338 | Section 05 | Step 5.1
Planning Conflict & Intervention Intelligence - Input Contract Audit

Purpose:
- Verify frozen Section 02 and Section 04 outputs.
- Verify all 9,240 KMC grid IDs and analysis domains.
- Verify final Conservation Priority = frozen GI * frozen UP.
- Audit actual Development Suitability / expert AHP readiness.
- Keep provisional suitability BLOCKED for final conflict scoring.
- Create a reproducible source registry and QA report.

No ML training, weights, scores, or source files are modified.
"""

from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd
import pyogrio


ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data/processed"
REPORTS = ROOT / "outputs/reports"
CI = PROCESSED / "conservation_intelligence"

FILES = {
    "pressure": (
        PROCESSED / "urbanization_pressure/inference/"
        "14c_2026_2028_urbanization_pressure_scores.csv"
    ),
    "eligibility": (
        PROCESSED / "urbanization_pressure/inference/"
        "14b_2025_inference_eligibility.csv"
    ),
    "green_importance": (
        CI / "frozen_importance/"
        "17j_frozen_green_importance_100m.csv"
    ),
    "gi_gpkg": (
        CI / "frozen_importance/"
        "17j_frozen_green_importance_100m.gpkg"
    ),
    "gi_manifest": (
        REPORTS / "conservation_intelligence/"
        "17j_green_importance_freeze_manifest.json"
    ),
    "conservation_priority": (
        CI / "conservation_priority/"
        "17m_primary_conservation_priority_100m.csv"
    ),
    "cp_gpkg": (
        CI / "conservation_priority/"
        "17m_primary_conservation_priority_100m.gpkg"
    ),
    "cp_qa": (
        REPORTS / "conservation_intelligence/"
        "17m_primary_conservation_priority_qa.json"
    ),
    "historical": (
        CI / "historical_threat/"
        "17e_historical_green_loss_exposure_100m.csv"
    ),
    "final_gis": (
        CI / "final_gis/"
        "17o_conservation_intelligence_final.gpkg"
    ),
    "final_qa": (
        REPORTS / "conservation_intelligence/"
        "17o_conservation_final_qa.json"
    ),
    "provisional_suitability": (
        PROCESSED / "development_suitability/"
        "weighted_suitability/"
        "03f_provisional_weighted_suitability_100m.csv"
    ),
    "provisional_ahp_qa": (
        REPORTS / "development_suitability/"
        "03e_provisional_ahp_audit.json"
    ),
    "provisional_suitability_qa": (
        REPORTS / "development_suitability/"
        "03f_provisional_weighted_suitability_qa.json"
    ),
    "final_ahp_status": (
        REPORTS / "16e_01_ahp_weighted_status.json"
    ),
    "ahp_sensitivity_status": (
        REPORTS / "16f_01_ahp_sensitivity_status.json"
    ),
    "ahp_pairwise_template": (
        REPORTS / "16c_l_ahp_pairwise_template.csv"
    ),
}

OUT_DIR = REPORTS / "planning_conflict_intervention"

OUT_REGISTRY = OUT_DIR / "17a_input_contract_registry.csv"
OUT_QA = OUT_DIR / "17a_input_contract_audit.json"

GI_FIELD = "green_importance_index_0_1"
UP_FIELD = "urbanization_pressure_score"
CP_FIELD = "primary_conservation_priority_0_1"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path):
    return json.loads(
        path.read_text(encoding="utf-8")
    )


def read_grid_csv(path, expected_rows):
    df = pd.read_csv(path)

    if (
        "grid_id" not in df.columns
        or len(df) != expected_rows
        or df.grid_id.isna().any()
        or not df.grid_id.is_unique
    ):
        raise ValueError(
            f"Grid identity failure: {path.name}"
        )

    return df.set_index("grid_id")


def numeric(df, column):
    if column not in df.columns:
        raise ValueError(
            f"Required field missing: {column}"
        )
    return pd.to_numeric(
        df[column],
        errors="coerce",
    ).to_numpy(dtype=float)


def main():

    print("=" * 78)
    print(
        "SECTION 05 | STEP 5.1 - "
        "FROZEN INPUT CONTRACT AUDIT"
    )
    print("=" * 78)

    for name, path in FILES.items():
        if not path.is_file():
            raise FileNotFoundError(
                f"{name}: {path}"
            )

    for path in (OUT_REGISTRY, OUT_QA):
        if path.exists():
            raise FileExistsError(
                f"Audit output already exists: {path}"
            )

    checks = {}

    def check(name, passed):
        checks[name] = bool(passed)
        state = "PASS" if passed else "FAIL"
        print(f"[{state}] {name}")

    def gate(section):
        if not all(checks.values()):
            raise SystemExit(
                f"{section}: input audit failed. "
                "No audit outputs created."
            )

    # ==============================================
    # 1. VERIFY SOURCE RELEASES
    # ==============================================

    print("\n[1] VERIFY FROZEN SOURCE RELEASES")

    manifest = read_json(FILES["gi_manifest"])
    cp_qa = read_json(FILES["cp_qa"])
    final_qa = read_json(FILES["final_qa"])

    check(
        "green_importance_frozen_release",
        manifest.get("status")
        == (
            "PASS_GREEN_IMPORTANCE_FROZEN_BASELINE_"
            "SENSITIVITY_PENDING"
        ),
    )

    check(
        "primary_conservation_priority_release",
        cp_qa.get("status")
        == (
            "PASS_MULTIPLICATIVE_CONSERVATION_PRIORITY_"
            "SENSITIVITY_PENDING"
        ),
    )

    check(
        "final_conservation_gis_release",
        final_qa.get("status")
        == (
            "PASS_FINAL_GIS_PACKAGE_WITH_DOCUMENTED_"
            "LIMITATIONS"
        ),
    )

    for kind, name in (
        ("csv", "green_importance"),
        ("gpkg", "gi_gpkg"),
    ):
        check(
            f"frozen_GI_{kind}_hash",
            sha256(FILES[name])
            == manifest["frozen_outputs"][kind]["sha256"],
        )

    for kind, name in (
        ("csv", "conservation_priority"),
        ("gpkg", "cp_gpkg"),
    ):
        check(
            f"frozen_CP_{kind}_hash",
            sha256(FILES[name])
            == cp_qa["output_sha256"][kind],
        )

    check(
        "final_GIS_package_hash",
        sha256(FILES["final_gis"])
        == final_qa["GIS_package"]["sha256"],
    )

    check(
        "original_pressure_source_unchanged",
        sha256(FILES["pressure"])
        == cp_qa["source_sha256"][
            "original_pressure_csv"
        ],
    )

    check(
        "historical_evidence_unchanged",
        sha256(FILES["historical"])
        == final_qa["source_sha256"][
            "historical_source_csv"
        ],
    )

    gate("Release integrity")

    # ==============================================
    # 2. GRID AND DOMAIN CONTRACTS
    # ==============================================

    print("\n[2] VERIFY 100m GRID AND MODEL DOMAINS")

    pressure = read_grid_csv(
        FILES["pressure"], 4620
    )

    eligibility = read_grid_csv(
        FILES["eligibility"], 9240
    )

    gi = read_grid_csv(
        FILES["green_importance"], 9240
    )

    cp = read_grid_csv(
        FILES["conservation_priority"], 9240
    )

    history = read_grid_csv(
        FILES["historical"], 9240
    )

    ids = gi.index

    check(
        "all_complete_grid_ID_sets_match",
        set(ids)
        == set(eligibility.index)
        == set(cp.index)
        == set(history.index),
    )

    check(
        "pressure_IDs_subset_of_KMC_grid",
        set(pressure.index).issubset(set(ids)),
    )

    gate("Grid alignment")

    eligibility = eligibility.loc[ids]
    cp = cp.loc[ids]
    history = history.loc[ids]

    flag_text = (
        eligibility.final_inference_ready
        .astype(str)
        .str.strip()
        .str.lower()
    )

    check(
        "eligibility_boolean_values_valid",
        flag_text.isin(
            ["true", "false", "1", "0"]
        ).all(),
    )

    ready_ids = set(
        eligibility.index[
            flag_text.isin(["true", "1"])
        ]
    )

    check(
        "pressure_scores_match_eligible_domain",
        ready_ids == set(pressure.index),
    )

    check(
        "relative_pressure_metadata",
        pressure.score_interpretation.eq(
            "MODEL_DERIVED_PRESSURE_SCORE_"
            "NOT_CALIBRATED_PROBABILITY"
        ).all(),
    )

    check(
        "pressure_2025_origin_2026_2028",
        pd.to_numeric(
            pressure.origin_year,
            errors="coerce",
        ).eq(2025).all()
        and pressure.pressure_period.eq(
            "2026-2028"
        ).all(),
    )

    p = numeric(pressure, UP_FIELD)

    check(
        "pressure_scores_valid_0_1",
        np.isfinite(p).all()
        and np.all((p >= 0) & (p <= 1)),
    )

    check(
        "pressure_threshold_0715_preserved",
        np.allclose(
            numeric(
                pressure, "operating_threshold"
            ),
            0.715,
            rtol=0,
            atol=1e-12,
        ),
    )

    check(
        "pressure_255_threshold_flags",
        int(
            numeric(
                pressure, "threshold_flag"
            ).sum()
        ) == 255,
    )

    gi_ready = gi.green_importance_status.eq(
        "GREEN_IMPORTANCE_CALCULATED"
    ).to_numpy()

    model_domain = ids.isin(pressure.index)

    cp_ready = cp.primary_priority_status.eq(
        "FUTURE_CONSERVATION_PRIORITY_CALCULATED"
    ).to_numpy()

    check(
        "green_importance_8727_scored",
        int(gi_ready.sum()) == 8727,
    )

    check(
        "common_domain_4609_cells",
        int(cp_ready.sum()) == 4609
        and np.array_equal(
            cp_ready,
            gi_ready & model_domain,
        ),
    )

    check(
        "GI_only_outside_pressure_4118",
        int(
            (gi_ready & ~model_domain).sum()
        ) == 4118,
    )

    check(
        "CP_unscored_4631_not_zero",
        int(cp[CP_FIELD].isna().sum()) == 4631
        and cp.loc[
            ~cp_ready, CP_FIELD
        ].isna().all(),
    )

    frozen_gi = numeric(gi, GI_FIELD)

    cp_gi = numeric(cp, GI_FIELD)

    cp_up = numeric(
        cp, "future_urbanization_pressure_score"
    )

    cp_value = numeric(cp, CP_FIELD)

    check(
        "GI_values_unchanged",
        np.allclose(
            frozen_gi,
            cp_gi,
            equal_nan=True,
            rtol=0,
            atol=0,
        ),
    )

    aligned_p = pressure.reindex(ids)

    check(
        "future_pressure_values_unchanged",
        np.allclose(
            cp_up,
            numeric(aligned_p, UP_FIELD),
            equal_nan=True,
            rtol=0,
            atol=1e-12,
        ),
    )

    check(
        "CP_reproduces_frozen_GI_times_UP",
        np.allclose(
            cp_value[cp_ready],
            cp_gi[cp_ready] * cp_up[cp_ready],
            atol=1e-12,
            rtol=0,
        ),
    )

    check(
        "CP_not_calculated_outside_valid_domain",
        np.isnan(
            cp_value[~cp_ready]
        ).all(),
    )

    check(
        "historical_robust_250m_available",
        np.isfinite(
            numeric(
                history,
                "historical_g2b_robust_exposure_share_250m",
            )
        ).all(),
    )

    gate("Domain and formula integrity")

    print("Total KMC cells:", len(ids))
    print("Pressure predictions:", len(pressure))
    print("GI scored:", int(gi_ready.sum()))
    print("Primary CP scored:", int(cp_ready.sum()))
    print(
        "GI scored, pressure NOT_IN_DOMAIN:",
        int((gi_ready & ~model_domain).sum()),
    )

    # ==============================================
    # 3. GIS PACKAGE CONTRACT
    # ==============================================

    print("\n[3] VERIFY FINAL GIS LAYERS")

    required_layers = {
        "primary_conservation_priority": 9240,
        "frozen_green_importance": 9240,
        "historical_loss_evidence": 9240,
        "priority_rank_sensitivity": 4609,
    }

    layers = set(
        pyogrio.list_layers(
            FILES["final_gis"]
        )[:, 0]
    )

    check(
        "four_final_GIS_layers_present",
        layers == set(required_layers),
    )

    for layer, count in required_layers.items():

        info = pyogrio.read_info(
            FILES["final_gis"],
            layer=layer,
        )

        check(
            f"{layer}_count",
            info["features"] == count,
        )

        check(
            f"{layer}_EPSG32644",
            info["crs"] == "EPSG:32644",
        )

    gate("GIS release")

    # ==============================================
    # 4. DEVELOPMENT SUITABILITY READINESS
    # ==============================================

    print("\n[4] AUDIT DEVELOPMENT SUITABILITY / AHP")

    ahp = read_json(
        FILES["provisional_ahp_qa"]
    )

    provisional_qa = read_json(
        FILES["provisional_suitability_qa"]
    )

    final_ahp = read_json(
        FILES["final_ahp_status"]
    )

    ahp_sensitivity = read_json(
        FILES["ahp_sensitivity_status"]
    )

    provisional = pd.read_csv(
        FILES["provisional_suitability"]
    )

    check(
        "provisional_suitability_grid_identity",
        "grid_id" in provisional.columns
        and len(provisional) == 9240
        and provisional.grid_id.is_unique
        and set(provisional.grid_id) == set(ids),
    )

    check(
        "provisional_AHP_labeled_correctly",
        ahp.get("status")
        == "NUMERICALLY_CONSISTENT_PROVISIONAL_ONLY",
    )

    check(
        "provisional_suitability_labeled_correctly",
        provisional_qa.get("status")
        == "PASS_PROVISIONAL",
    )

    check(
        "expert_AHP_blocker_documented",
        final_ahp.get("status")
        == "BLOCKED_AWAITING_VALID_EXPERT_AHP",
    )

    check(
        "final_AHP_sensitivity_blocker_documented",
        ahp_sensitivity.get("status")
        == "BLOCKED_AWAITING_AHP_WEIGHTED_MODEL",
    )

    ratio = ahp.get(
        "consistency_ratio"
    )

    check(
        "provisional_CR_numerically_below_010",
        ratio is not None
        and np.isfinite(float(ratio))
        and 0 <= float(ratio) < 0.10,
    )

    gate("AHP provenance")

    print("Provisional AHP CR:", ratio)
    print(
        "Expert AHP final status:",
        final_ahp["status"],
    )

    print(
        "Final Development Suitability usable: NO"
    )

    print(
        "Final DCI calculation allowed: NO"
    )

    # ==============================================
    # 5. WRITE INPUT CONTRACT REGISTRY
    # ==============================================

    print("\n[5] WRITE SECTION 05 INPUT CONTRACT")

    registry_spec = {
        "pressure": (
            4620, "FROZEN_RELATIVE_SCORE", True
        ),
        "eligibility": (
            9240, "FROZEN_DOMAIN_CONTRACT", True
        ),
        "green_importance": (
            9240, "FROZEN_GREEN_IMPORTANCE", True
        ),
        "conservation_priority": (
            9240, "FROZEN_PRIMARY_CP", True
        ),
        "historical": (
            9240, "SUPPORTING_EVIDENCE", True
        ),
        "final_gis": (
            9240, "FINAL_GIS_FOUR_LAYERS", True
        ),
        "provisional_suitability": (
            9240, "PROVISIONAL_NOT_FINAL", False
        ),
        "provisional_ahp_qa": (
            None, "PROVISIONAL_AHP_EVIDENCE", False
        ),
        "final_ahp_status": (
            None, "BLOCKED_EXPERT_AHP", False
        ),
        "ahp_sensitivity_status": (
            None, "BLOCKED_FINAL_SENSITIVITY", False
        ),
    }

    registry = pd.DataFrame([
        {
            "input_name": name,
            "path": FILES[name].relative_to(
                ROOT
            ).as_posix(),
            "sha256": sha256(FILES[name]),
            "expected_records": count,
            "source_status": status,
            "usable_for_final_section05": usable,
        }
        for name, (count, status, usable)
        in registry_spec.items()
    ])

    report = {
        "section": "05_planning_conflict_intervention",
        "step": "5.1",
        "status": (
            "PASS_INPUT_AUDIT_WITH_EXPERT_AHP_BLOCKER"
        ),
        "KMC_grid_cells": 9240,
        "grid_crs": "EPSG:32644",
        "available_frozen_inputs": {
            "urbanization_pressure": True,
            "pressure_domain_eligible_cells": 4620,
            "green_importance": True,
            "green_importance_scored_cells": 8727,
            "conservation_priority": True,
            "conservation_priority_scored_cells": 4609,
            "historical_loss_evidence": True,
            "final_conservation_gis": True,
        },
        "development_suitability": {
            "provisional_output_exists": True,
            "provisional_consistency_ratio": float(ratio),
            "provisional_ahp_status": ahp["status"],
            "final_ahp_status": final_ahp["status"],
            "final_ahp_sensitivity_status": (
                ahp_sensitivity["status"]
            ),
            "verified_final_expert_ahp": False,
            "permitted_for_final_DCI": False,
            "must_not_be_silently_promoted": True,
        },
        "domain_rules": {
            "pressure_out_of_domain": (
                "Missing; never substitute zero"
            ),
            "primary_CP_formula": (
                "Frozen GI multiplied by frozen UP"
            ),
            "CP_out_of_domain": (
                "Missing; never substitute zero"
            ),
            "historical_loss": (
                "Supporting evidence only"
            ),
            "provisional_UDA": (
                "Planning context only; not legal zoning"
            ),
        },
        "step_52_common_grid_preparation_allowed": True,
        "final_development_conflict_allowed": False,
        "final_intervention_scoring_allowed": False,
        "checks": checks,
        "source_registry": (
            "17a_input_contract_registry.csv"
        ),
        "sources_modified": False,
    }

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    registry.to_csv(
        OUT_REGISTRY,
        index=False,
    )

    OUT_QA.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print("Registry:", OUT_REGISTRY)
    print("QA report:", OUT_QA)

    print("\n" + "=" * 78)
    print(
        "STEP 5.1: "
        "PASS_INPUT_AUDIT_WITH_EXPERT_AHP_BLOCKER"
    )
    print(
        "Frozen pressure / GI / CP verified: YES"
    )
    print(
        "Development Suitability final AHP: BLOCKED"
    )
    print(
        "Common-grid preparation may proceed: YES"
    )
    print(
        "Final conflict/intervention scoring: BLOCKED"
    )
    print(
        "Source files modified: NO"
    )
    print("=" * 78)


if __name__ == "__main__":
    main()
