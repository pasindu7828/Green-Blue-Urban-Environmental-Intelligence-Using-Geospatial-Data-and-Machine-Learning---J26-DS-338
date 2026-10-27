
"""
J26-DS-338 | Section 04, Stage 4.11
Verify and freeze the independent Green Importance research baseline.

The frozen snapshot has the SAME numerical scores, criterion contributions,
weights, 9,240 IDs and geometries as the validated Stage 4.10 output.
Only the metadata flag green_importance_frozen changes from False to True.

Frozen means reproducible baseline, NOT final ecological validation.
Stage 4.15 sensitivity remains mandatory.

Inputs: 17g criteria, 17h CRITIC weights, 17i index + their QA files.
Outputs: one frozen CSV, one frozen GPKG, one freeze manifest JSON.
No historical threat, future pressure, suitability, SHAP or zoning is used.
Never overwrite a previously frozen output or modify original sources.
"""

from pathlib import Path
import hashlib
import json

import geopandas as gpd
import numpy as np
import pandas as pd
import pyogrio


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data/processed/conservation_intelligence"
REPORT = ROOT / "outputs/reports/conservation_intelligence"

CRITERIA_FILE = DATA / (
    "importance_criteria/"
    "17g_green_importance_criteria_standardized_100m.csv"
)

WEIGHTS_FILE = REPORT / "17h_critic_weights.csv"

INDEX_CSV = DATA / (
    "importance_index/17i_green_importance_index_100m.csv"
)

INDEX_GPKG = DATA / (
    "importance_index/17i_green_importance_index_100m.gpkg"
)

QA_FILES = {
    "criteria": (
        REPORT / "17g_green_importance_criteria_qa.json",
        "PASS_STANDARDIZED_CANDIDATES_WEIGHTING_PENDING",
    ),
    "weights": (
        REPORT / "17h_critic_weighting_qa.json",
        "PASS_CRITIC_WEIGHTING_INDEX_PENDING",
    ),
    "index": (
        REPORT / "17i_green_importance_index_qa.json",
        "PASS_GREEN_IMPORTANCE_INDEX_NOT_FROZEN",
    ),
}

FROZEN_DIR = DATA / "frozen_importance"

FROZEN_CSV = (
    FROZEN_DIR / "17j_frozen_green_importance_100m.csv"
)

FROZEN_GPKG = (
    FROZEN_DIR / "17j_frozen_green_importance_100m.gpkg"
)

MANIFEST = (
    REPORT / "17j_green_importance_freeze_manifest.json"
)

FIELDS = {
    "G1": "G1_current_green_0_1",
    "G2": "G2_historical_persistence_0_1",
    "G3": "G3_local_green_context_0_1",
}

CODES = list(FIELDS)
SCORE = "green_importance_index_0_1"
TOL = 1e-10


def sha256(path):

    h = hashlib.sha256()

    with path.open("rb") as file:
        for block in iter(
            lambda: file.read(1024 * 1024),
            b"",
        ):
            h.update(block)

    return h.hexdigest()


def numeric(frame, columns):

    return (
        frame[columns]
        .apply(pd.to_numeric, errors="coerce")
        .to_numpy(dtype=float)
    )


def main():

    print("=" * 76)
    print(
        "SECTION 04 | STAGE 4.11 - "
        "GREEN IMPORTANCE VERIFICATION AND FREEZE"
    )
    print("=" * 76)

    required = [
        CRITERIA_FILE,
        WEIGHTS_FILE,
        INDEX_CSV,
        INDEX_GPKG,
    ] + [
        entry[0] for entry in QA_FILES.values()
    ]

    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)

    checks = {}

    def check(label, condition):

        passed = bool(condition)
        checks[label] = passed

        print(
            f"[{'PASS' if passed else 'FAIL'}] {label}"
        )

    def gate(phase):

        if not all(checks.values()):
            raise SystemExit(
                f"{phase}: QA failed. "
                "Freeze was not performed."
            )

    # ==================================================
    # 1. VERIFY UPSTREAM STAGES AND SOURCE HASHES
    # ==================================================

    print(
        "\n[1] CHECK UPSTREAM STATUS "
        "AND SOURCE FINGERPRINTS"
    )

    audits = {}

    for key, (path, expected) in QA_FILES.items():

        audit = json.loads(
            path.read_text(encoding="utf-8")
        )

        audits[key] = audit

        check(
            f"stage_{key}_passed",
            audit.get("status") == expected,
        )

    check(
        "stage_49_critic_method",
        audits["weights"].get("method")
        == "CRITIC_OBJECTIVE_WEIGHTING",
    )

    check(
        "stage_410_not_yet_frozen",
        audits["index"].get(
            "green_importance_frozen"
        ) is False,
    )

    check(
        "criteria_hash_matches_both_audits",
        sha256(CRITERIA_FILE)
        == audits["weights"].get("input_csv_sha256")
        == audits["index"].get("source_criteria_sha256"),
    )

    check(
        "weight_file_hash_matches_stage_410",
        sha256(WEIGHTS_FILE)
        == audits["index"].get("source_weights_sha256"),
    )

    check(
        "prior_index_has_no_threat_or_priority",
        audits["index"].get("historical_loss_used") is False
        and audits["index"].get("future_pressure_used") is False
        and audits["index"].get(
            "development_suitability_used"
        ) is False
        and audits["index"].get(
            "conservation_priority_calculated"
        ) is False,
    )

    gate("Upstream")

    # ==================================================
    # 2. REPRODUCE GREEN IMPORTANCE NUMERICALLY
    # ==================================================

    print("\n[2] REPRODUCE SCORES AND CONTRIBUTIONS")

    df = pd.read_csv(INDEX_CSV)
    weights_df = pd.read_csv(WEIGHTS_FILE)

    required_columns = {
        "grid_id",
        "green_importance_status",
        "green_importance_frozen",
        "current_green_condition_status",
        "kmc_overlap_area_m2",
        SCORE,
        "largest_weighted_contribution",
        *FIELDS.values(),
    }

    required_columns.update(
        f"{code}_weighted_contribution"
        for code in CODES
    )

    required_columns.update(
        f"{code}_critic_weight"
        for code in CODES
    )

    if not required_columns.issubset(df.columns):
        raise ValueError(
            "Missing index columns: "
            f"{sorted(required_columns - set(df.columns))}"
        )

    weight_columns = {
        "criterion",
        "source_field",
        "critic_weight",
    }

    if not weight_columns.issubset(weights_df.columns):
        raise ValueError(
            "Missing CRITIC weight fields"
        )

    check(
        "index_has_9240_unique_ids",
        len(df) == 9240
        and df.grid_id.notna().all()
        and df.grid_id.is_unique,
    )

    check(
        "weight_table_three_correct_criteria",
        len(weights_df) == 3
        and weights_df.criterion.is_unique
        and set(weights_df.criterion) == set(CODES),
    )

    gate("Source format")

    weights_df = (
        weights_df
        .set_index("criterion")
        .loc[CODES]
    )

    weights = pd.to_numeric(
        weights_df.critic_weight,
        errors="coerce",
    ).to_numpy(dtype=float)

    check(
        "weights_match_fields_and_sum_to_one",
        all(
            weights_df.loc[code, "source_field"]
            == FIELDS[code]
            for code in CODES
        )
        and np.isfinite(weights).all()
        and (weights >= 0).all()
        and np.isclose(
            weights.sum(),
            1.0,
            atol=1e-12,
        ),
    )

    check(
        "weights_match_stage_410_qa",
        all(
            np.isclose(
                weights[i],
                audits["index"]
                .get("weights", {})
                .get(code, np.nan),
                atol=1e-12,
            )
            for i, code in enumerate(CODES)
        ),
    )

    status = df.green_importance_status.astype(str)

    ready = status.eq(
        "GREEN_IMPORTANCE_CALCULATED"
    ).to_numpy()

    no_green = status.eq(
        "NO_CURRENT_GREEN_DETECTED"
    ).to_numpy()

    unknown = status.eq(
        "UNKNOWN_CURRENT_GREEN_CONDITION"
    ).to_numpy()

    check(
        "population_partition_8727_468_45",
        int(ready.sum()) == 8727
        and int(no_green.sum()) == 468
        and int(unknown.sum()) == 45
        and (ready | no_green | unknown).all(),
    )

    check(
        "original_index_unfrozen",
        df.green_importance_frozen
        .astype("string")
        .str.lower()
        .eq("false")
        .all(),
    )

    gate("Weight and population")

    X = numeric(
        df.loc[ready],
        list(FIELDS.values()),
    )

    contribution_columns = [
        f"{code}_weighted_contribution"
        for code in CODES
    ]

    contributions = numeric(
        df.loc[ready],
        contribution_columns,
    )

    GI = pd.to_numeric(
        df.loc[ready, SCORE],
        errors="coerce",
    ).to_numpy(dtype=float)

    all_scores = pd.to_numeric(
        df[SCORE],
        errors="coerce",
    ).to_numpy(dtype=float)

    check(
        "all_ready_criteria_finite_0_1",
        np.isfinite(X).all()
        and (
            (X >= -TOL)
            & (X <= 1 + TOL)
        ).all(),
    )

    check(
        "index_reproduces_weighted_formula",
        np.isfinite(GI).all()
        and np.allclose(
            GI,
            X @ weights,
            rtol=0,
            atol=TOL,
        ),
    )

    check(
        "all_component_contributions_reproduced",
        np.isfinite(contributions).all()
        and np.allclose(
            contributions,
            X * weights,
            rtol=0,
            atol=TOL,
        ),
    )

    check(
        "weight_columns_identical_to_frozen_critic",
        all(
            np.allclose(
                pd.to_numeric(
                    df[f"{code}_critic_weight"],
                    errors="coerce",
                ),
                weights[i],
                rtol=0,
                atol=1e-12,
            )
            for i, code in enumerate(CODES)
        ),
    )

    check(
        "index_0_to_1",
        (
            (GI >= -TOL)
            & (GI <= 1 + TOL)
        ).all(),
    )

    check(
        "513_unscored_remain_nan",
        np.isnan(
            all_scores[~ready]
        ).all(),
    )

    check(
        "unknown_and_nongreen_source_statuses_consistent",
        df.loc[
            unknown,
            "current_green_condition_status",
        ].eq("UNKNOWN_NO_VALID_LULC").all()
        and df.loc[
            no_green,
            "current_green_condition_status",
        ].eq("NO_GREEN_DETECTED_IN_VALID_AREA").all(),
    )

    check(
        "dominant_contribution_reproduced",
        np.array_equal(
            df.loc[
                ready,
                "largest_weighted_contribution",
            ].astype(str).to_numpy(),
            np.asarray(CODES)[
                np.argmax(contributions, axis=1)
            ],
        ),
    )

    check(
        "index_distribution_matches_stage_410",
        np.isclose(
            np.mean(GI),
            audits["index"]["index_distribution"]["mean"],
            atol=1e-10,
        )
        and np.isclose(
            np.median(GI),
            audits["index"]["index_distribution"]["median"],
            atol=1e-10,
        ),
    )

    gate("Numerical reproducibility")

    # ==================================================
    # 3. VERIFY CSV AND GIS CONSISTENCY
    # ==================================================

    print("\n[3] CROSS-CHECK CSV AND GIS")

    geo = gpd.read_file(INDEX_GPKG)

    check(
        "gis_9240_unique_matching_ids",
        len(geo) == 9240
        and geo.grid_id.notna().all()
        and geo.grid_id.is_unique
        and set(geo.grid_id) == set(df.grid_id),
    )

    check(
        "gis_epsg32644_valid",
        geo.crs is not None
        and geo.crs.to_epsg() == 32644
        and geo.geometry.notna().all()
        and geo.geometry.is_valid.all(),
    )

    gate("GIS identity")

    geo = (
        geo.set_index("grid_id")
        .loc[df.grid_id]
        .reset_index()
    )

    gis_scores = pd.to_numeric(
        geo[SCORE],
        errors="coerce",
    ).to_numpy(dtype=float)

    area = pd.to_numeric(
        df.kmc_overlap_area_m2,
        errors="coerce",
    ).to_numpy(dtype=float)

    check(
        "gis_index_matches_csv",
        np.allclose(
            gis_scores,
            all_scores,
            rtol=0,
            atol=TOL,
            equal_nan=True,
        ),
    )

    check(
        "gis_criteria_match_csv",
        np.allclose(
            numeric(geo, list(FIELDS.values())),
            numeric(df, list(FIELDS.values())),
            rtol=0,
            atol=TOL,
            equal_nan=True,
        ),
    )

    check(
        "gis_contributions_match_csv",
        np.allclose(
            numeric(geo, contribution_columns),
            numeric(df, contribution_columns),
            rtol=0,
            atol=TOL,
            equal_nan=True,
        ),
    )

    check(
        "gis_status_matches_csv",
        np.array_equal(
            geo.green_importance_status
            .astype(str)
            .to_numpy(),
            status.to_numpy(),
        ),
    )

    check(
        "gis_clipped_areas_match_csv",
        np.isfinite(area).all()
        and (area > 0).all()
        and np.max(
            np.abs(
                geo.geometry.area.to_numpy(dtype=float)
                - area
            )
        ) <= 1.0,
    )

    forbidden = [
        "pressure",
        "shap",
        "suitability",
        "zoning",
        "priority",
        "historical_g2b",
        "historical_loss",
    ]

    check(
        "no_threat_priority_or_suitability_fields",
        not any(
            token in column.lower()
            for column in df.columns
            for token in forbidden
        ),
    )

    gate("GIS and conceptual independence")

    # ==================================================
    # 4. CREATE THE FROZEN SNAPSHOT
    # ==================================================

    print("\n[4] CREATE ONE-TIME FROZEN SNAPSHOT")

    # Never silently overwrite an existing frozen output.
    for path in [
        FROZEN_CSV,
        FROZEN_GPKG,
        MANIFEST,
    ]:
        if path.exists():
            raise FileExistsError(
                f"Freeze output already exists: {path}. "
                "Do not overwrite a frozen release; "
                "inspect the existing snapshot."
            )

    frozen_df = df.copy()

    frozen_df["green_importance_frozen"] = True

    frozen_geo = geo.copy()

    frozen_geo["green_importance_frozen"] = True

    FROZEN_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORT.mkdir(
        parents=True,
        exist_ok=True,
    )

    frozen_df.to_csv(
        FROZEN_CSV,
        index=False,
    )

    frozen_geo.to_file(
        FROZEN_GPKG,
        layer="frozen_green_importance_100m",
        driver="GPKG",
    )

    # ==================================================
    # 5. VERIFY THE EXPORTED SNAPSHOT
    # ==================================================

    print("\n[5] VERIFY FROZEN OUTPUTS")

    reread = pd.read_csv(FROZEN_CSV)
    frozen_gis = gpd.read_file(FROZEN_GPKG)
    gis_info = pyogrio.read_info(FROZEN_GPKG)

    check(
        "frozen_csv_9240_matching_scores",
        len(reread) == 9240
        and reread.grid_id.is_unique
        and np.allclose(
            pd.to_numeric(
                reread[SCORE],
                errors="coerce",
            ),
            all_scores,
            equal_nan=True,
            rtol=0,
            atol=TOL,
        ),
    )

    check(
        "frozen_gis_9240",
        gis_info["features"] == 9240
        and len(frozen_gis) == 9240
        and frozen_gis.grid_id.is_unique,
    )

    check(
        "frozen_gis_scores_reproduced",
        np.allclose(
            pd.to_numeric(
                frozen_gis
                .set_index("grid_id")
                .loc[df.grid_id, SCORE],
                errors="coerce",
            ).to_numpy(dtype=float),
            all_scores,
            equal_nan=True,
            rtol=0,
            atol=TOL,
        ),
    )

    check(
        "frozen_gis_flag_true",
        frozen_gis.green_importance_frozen
        .astype("string")
        .str.lower()
        .eq("true")
        .all(),
    )

    check(
        "frozen_flag_true",
        reread.green_importance_frozen
        .astype("string")
        .str.lower()
        .eq("true")
        .all(),
    )

    gate("Frozen export verification")

    # ==================================================
    # 6. WRITE THE FREEZE MANIFEST
    # ==================================================

    print("\n[6] WRITE FREEZE MANIFEST")

    manifest = {
        "stage": "4.11",

        "status": (
            "PASS_GREEN_IMPORTANCE_FROZEN_BASELINE_"
            "SENSITIVITY_PENDING"
        ),

        "freeze_type": (
            "REPRODUCIBLE_BASELINE_"
            "NOT_FINAL_ECOLOGICAL_VALIDATION"
        ),

        "method": "CRITIC_WEIGHTED_ADDITIVE",

        "study_area": "Kaduwela Municipal Council",

        "crs": "EPSG:32644",

        "grid_cells": 9240,

        "scored_current_green_cells": 8727,

        "no_green_detected_unscored": 468,

        "unknown_green_condition_unscored": 45,

        "criteria": FIELDS,

        "exact_critic_weights": {
            code: float(weights[i])
            for i, code in enumerate(CODES)
        },

        "input_sha256": {
            "standardized_criteria_csv":
                sha256(CRITERIA_FILE),

            "critic_weights_csv":
                sha256(WEIGHTS_FILE),

            "unfrozen_index_csv":
                sha256(INDEX_CSV),

            "unfrozen_index_gpkg":
                sha256(INDEX_GPKG),
        },

        "frozen_outputs": {
            "csv": {
                "path": str(
                    FROZEN_CSV.relative_to(ROOT)
                ).replace("\\", "/"),

                "sha256": sha256(FROZEN_CSV),
            },

            "gpkg": {
                "path": str(
                    FROZEN_GPKG.relative_to(ROOT)
                ).replace("\\", "/"),

                "sha256": sha256(FROZEN_GPKG),
            },
        },

        "freeze_only_attribute_change": (
            "green_importance_frozen: False -> True"
        ),

        "index_summary": {
            "min": float(GI.min()),
            "mean": float(GI.mean()),
            "median": float(np.median(GI)),
            "max": float(GI.max()),
        },

        "largest_contribution_counts": {
            code: int(
                np.sum(
                    df.loc[
                        ready,
                        "largest_weighted_contribution",
                    ].to_numpy() == code
                )
            )
            for code in CODES
        },

        "checks": checks,

        "historic_green_loss_in_importance": False,

        "future_urbanization_pressure_in_importance": False,

        "development_suitability_in_importance": False,

        "sensitivity_status": "PENDING_STAGE_4_15",

        "future_integration_rule": (
            "Use the frozen CSV/GPKG and verify both "
            "SHA256 hashes before integrating threats."
        ),

        "limitations": [
            (
                "This is a technical baseline freeze, "
                "not proof of ecological validity."
            ),
            (
                "Historical persistence is the largest "
                "weighted contribution in many cells."
            ),
            (
                "Future sensitivity analyses must compare "
                "alternatives without silently modifying "
                "this baseline."
            ),
            (
                "Frozen files are integrity-fingerprinted, "
                "not operating-system write-protected."
            ),
        ],
    }

    MANIFEST.write_text(
        json.dumps(
            manifest,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("Frozen CSV:", FROZEN_CSV)
    print("Frozen GPKG:", FROZEN_GPKG)
    print("Freeze manifest:", MANIFEST)

    print("\n" + "=" * 76)
    print(
        "STAGE 4.11: "
        "PASS_GREEN_IMPORTANCE_FROZEN_BASELINE_"
        "SENSITIVITY_PENDING"
    )
    print("Scored green cells: 8,727 | Unscored: 513")
    print("Threat integration: NOT YET PERFORMED")
    print("Sensitivity: PENDING")
    print("=" * 76)


if __name__ == "__main__":
    main()
