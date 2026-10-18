
"""
J26-DS-338 | Section 05 | Step 18E
HISTORICAL GREEN-LOSS VIGILANCE

Use frozen historical green-to-built observations as
supporting planning context.

Do NOT:
- Modify DCI or Conservation Priority.
- Treat historical conversion as a future prediction.
- Interpret conversion as proven illegal encroachment.
- Invent HIGH/MEDIUM/LOW risk thresholds.
- Fill missing historical evidence with zero.
"""

from pathlib import Path
from tempfile import TemporaryDirectory
import hashlib
import json

import geopandas as gpd
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

BASE = (
    ROOT / "data/processed/planning_conflict_intervention"
)

REPORT = (
    ROOT / "outputs/reports/planning_conflict_intervention"
)

INTEGRATED = (
    BASE / "common_grid"
    / "18b_common_grid_integration_100m.csv"
)

INTEGRATED_QA = (
    REPORT / "18b_common_grid_integration_qa.json"
)

TYPOLOGY = (
    BASE / "conflict_typology"
    / "18d_planning_conflict_typology_100m.csv"
)

TYPOLOGY_QA = (
    REPORT / "18d_planning_conflict_typology_qa.json"
)

FROZEN_GIS = (
    ROOT
    / "data/processed/conservation_intelligence/final_gis"
    / "17o_conservation_intelligence_final.gpkg"
)

HISTORY_LAYER = "historical_loss_evidence"

OUT_DIR = BASE / "historical_vigilance"

OUT_CSV = (
    OUT_DIR / "18e_historical_green_loss_vigilance_100m.csv"
)

OUT_QA = (
    REPORT / "18e_historical_green_loss_vigilance_qa.json"
)

N = 9240
DOMAIN_COUNT = 4620
COMMON_COUNT = 4609


def require(condition, label):

    ok = bool(condition)

    print(
        f"[{'PASS' if ok else 'FAIL'}] {label}"
    )

    if not ok:
        raise ValueError(
            f"Step 18E validation failed: {label}"
        )


def sha256(path):

    digest = hashlib.sha256()

    with path.open("rb") as stream:
        for block in iter(
            lambda: stream.read(1024 * 1024),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def number(df, field):

    if field not in df:
        raise KeyError(
            f"Missing field: {field}"
        )

    return pd.to_numeric(
        df[field],
        errors="coerce",
    ).to_numpy(dtype=float)


def validate_grid(df, label):

    require(
        len(df) == N
        and "grid_id" in df
        and df.grid_id.notna().all()
        and df.grid_id.is_unique,
        f"{label}: 9,240 unique nonmissing grid IDs",
    )


def optional_boolean(df, field):

    require(
        field in df,
        f"Boolean field present: {field}",
    )

    text = (
        df[field]
        .astype("string")
        .str.lower()
        .str.strip()
    )

    require(
        text.dropna().isin(
            ["true", "false", "1", "0"]
        ).all(),
        f"{field}: valid true/false/missing encoding",
    )

    output = pd.Series(
        pd.NA,
        index=df.index,
        dtype="boolean",
    )

    output.loc[
        text.isin(["true", "1"])
    ] = True

    output.loc[
        text.isin(["false", "0"])
    ] = False

    return output


def supported_boolean(present, supported):

    result = pd.Series(
        pd.NA,
        index=range(len(supported)),
        dtype="boolean",
    )

    result.loc[supported] = present[supported]

    return result


def main():

    print("=" * 78)
    print(
        "SECTION 05 | STEP 18E - "
        "HISTORICAL GREEN-LOSS VIGILANCE"
    )
    print("=" * 78)

    inputs = [
        INTEGRATED,
        INTEGRATED_QA,
        TYPOLOGY,
        TYPOLOGY_QA,
        FROZEN_GIS,
    ]

    for path in inputs:
        if not path.is_file():
            raise FileNotFoundError(path)

    for path in [OUT_CSV, OUT_QA]:
        if path.exists():
            raise FileExistsError(
                f"Refusing to overwrite: {path}"
            )

    hashes = {
        p.relative_to(ROOT).as_posix(): sha256(p)
        for p in inputs
    }

    # =====================================================
    # 1. VERIFY FROZEN SOURCE CONTRACTS
    # =====================================================

    print("\n[1] VERIFY FROZEN SOURCE CONTRACTS")

    qa_b = json.loads(
        INTEGRATED_QA.read_text(encoding="utf-8")
    )

    qa_d = json.loads(
        TYPOLOGY_QA.read_text(encoding="utf-8")
    )

    require(
        qa_b.get("status")
        == "PASS_COMMON_GRID_FINAL_EXPERT_AHP_INTEGRATION",
        "18B QA passed",
    )

    require(
        qa_d.get("status")
        == "PASS_EXPLORATORY_PLANNING_TYPOLOGY_WITH_DOMAIN_LIMITS",
        "18D QA passed",
    )

    require(
        qa_b["outputs"]["csv_sha256"]
        == sha256(INTEGRATED),
        "18B integrated CSV hash verified",
    )

    require(
        qa_d["output"]["sha256"]
        == sha256(TYPOLOGY),
        "18D typology CSV hash verified",
    )

    integrated_key = (
        INTEGRATED.relative_to(ROOT).as_posix()
    )

    require(
        qa_d["inputs_sha256"][integrated_key]
        == sha256(INTEGRATED),
        "18D uses this exact 18B integration",
    )

    conservation_key = (
        FROZEN_GIS.relative_to(ROOT).as_posix()
    )

    require(
        qa_b["input_sha256"][conservation_key]
        == sha256(FROZEN_GIS),
        "Historical evidence belongs to frozen "
        "18B conservation source",
    )

    # =====================================================
    # 2. LOAD AND ALIGN DATA
    # =====================================================

    print("\n[2] LOAD AND ALIGN GRID EVIDENCE")

    joined = pd.read_csv(INTEGRATED)

    types = pd.read_csv(TYPOLOGY)

    history = gpd.read_file(
        FROZEN_GIS,
        layer=HISTORY_LAYER,
    ).drop(columns="geometry")

    for name, frame in [
        ("18B", joined),
        ("18D", types),
        ("17O historical", history),
    ]:
        validate_grid(frame, name)

    require(
        set(joined.grid_id)
        == set(types.grid_id)
        == set(history.grid_id),
        "Grid IDs agree across all sources",
    )

    types = (
        types.set_index("grid_id")
        .loc[joined.grid_id]
        .reset_index()
    )

    history = (
        history.set_index("grid_id")
        .loc[joined.grid_id]
        .reset_index()
    )

    domain = (
        joined.pressure_domain_status_section05
        .eq("IN_MODEL_DOMAIN")
        .to_numpy(dtype=bool)
    )

    require(
        int(domain.sum()) == DOMAIN_COUNT,
        "4,620 pressure-domain cells preserved",
    )

    full_type = ~types.planning_evidence_type.isin([
        "OUTSIDE_PRESSURE_MODEL_DOMAIN",
        "DCI_AVAILABLE__FROZEN_CP_MISSING",
    ]).to_numpy(dtype=bool)

    require(
        int(full_type.sum()) == COMMON_COUNT,
        "4,609 complete typology cells preserved",
    )

    flagged = number(
        types,
        "pressure_threshold_flag_frozen",
    )

    require(
        int(np.count_nonzero(
            flagged[domain] == 1
        )) == 255
        and np.isnan(flagged[~domain]).all(),
        "255 frozen pressure flags preserved",
    )

    require(
        np.allclose(
            number(
                joined,
                "development_suitability_score",
            ),
            number(
                types,
                "development_suitability_score",
            ),
            atol=1e-12,
            rtol=0,
        ),
        "Development suitability preserved",
    )

    require(
        np.allclose(
            number(
                joined,
                "primary_conservation_priority_0_1",
            ),
            number(
                types,
                "conservation_conflict_index_0_1",
            ),
            equal_nan=True,
            atol=1e-12,
            rtol=0,
        ),
        "Frozen conservation indicator unchanged",
    )

    dual = optional_boolean(
        types,
        "joint_flagged_pressure_lower_suitability_upper_cp",
    )

    managed = optional_boolean(
        types,
        "preliminary_managed_growth_screening_candidate",
    )

    require(
        int(dual.fillna(False).sum()) == 7,
        "Seven preliminary dual-concern screens preserved",
    )

    require(
        int(managed.fillna(False).sum()) == 1,
        "One preliminary managed-growth screen preserved",
    )

    # =====================================================
    # 3. PREPARE HISTORICAL EVIDENCE
    # =====================================================

    print("\n[3] VALIDATE HISTORICAL EXPOSURE")

    output = pd.DataFrame({

        "grid_id":
            joined.grid_id,

        "kmc_overlap_area_m2":
            number(
                joined,
                "kmc_overlap_area_m2",
            ),

        "pressure_domain_status":
            joined.pressure_domain_status_section05,

        "planning_evidence_type":
            types.planning_evidence_type,

        "pressure_threshold_flag_frozen":
            pd.Series(flagged).astype("Int64"),

        "historical_loss_early_period":
            history.historical_loss_early_period,

        "historical_loss_late_period":
            history.historical_loss_late_period,

    })

    supports = {}
    detections = {}

    for mode in ("robust", "strict"):

        for radius in (250, 500):

            stem = f"historical_g2b_{mode}"

            valid_name = (
                f"{stem}_valid_pixels_{radius}m"
            )

            changed_name = (
                f"{stem}_converted_pixels_{radius}m"
            )

            share_name = (
                f"{stem}_exposure_share_{radius}m"
            )

            state_name = (
                f"{stem}_context_status_{radius}m"
            )

            for field in [
                valid_name,
                changed_name,
                share_name,
                state_name,
            ]:

                require(
                    field in history,
                    f"Frozen historical field exists: {field}",
                )

            valid = number(
                history,
                valid_name,
            )

            changed = number(
                history,
                changed_name,
            )

            share = number(
                history,
                share_name,
            )

            require(
                not np.isinf(valid).any()
                and not np.isinf(changed).any()
                and not np.isinf(share).any(),
                f"{mode} {radius}m: no infinite values",
            )

            require(
                np.all(
                    valid[np.isfinite(valid)] >= 0
                )
                and np.all(
                    changed[np.isfinite(changed)] >= 0
                ),
                f"{mode} {radius}m: "
                "nonnegative pixel counts",
            )

            both = (
                np.isfinite(valid)
                & np.isfinite(changed)
            )

            require(
                np.all(
                    changed[both] <= valid[both]
                ),
                f"{mode} {radius}m: converted pixels "
                "do not exceed valid pixels",
            )

            observed_share = np.isfinite(
                share
            )

            require(
                np.all(
                    (share[observed_share] >= -1e-12)
                    & (share[observed_share] <= 1 + 1e-12)
                ),
                f"{mode} {radius}m: "
                "exposure share in [0,1]",
            )

            supported = (
                both
                & (valid > 0)
                & observed_share
            )

            detected = (
                supported
                & (changed > 0)
            )

            key = (
                mode,
                radius,
            )

            supports[key] = supported
            detections[key] = detected

            for field in [
                valid_name,
                changed_name,
                share_name,
                state_name,
            ]:

                output[field] = history[field]

            output[
                f"historical_{mode}_detection_{radius}m"
            ] = supported_boolean(
                detected,
                supported,
            )

            if share_name in joined:

                require(
                    np.allclose(
                        share,
                        number(joined, share_name),
                        equal_nan=True,
                        atol=1e-12,
                        rtol=0,
                    ),
                    f"{mode} {radius}m: "
                    "historical share unchanged",
                )

            print(
                f"{mode.upper()} {radius}m: "
                f"supported={int(supported.sum())}, "
                f"conversion_detected={int(detected.sum())}"
            )

    # =====================================================
    # 4. HISTORICAL VIGILANCE CONTEXT
    # =====================================================

    print("\n[4] BUILD SUPPORTING HISTORICAL CONTEXT")

    robust_support = (
        supports[("robust", 250)]
        | supports[("robust", 500)]
    )

    strict_support = (
        supports[("strict", 250)]
        | supports[("strict", 500)]
    )

    robust_detected = (
        detections[("robust", 250)]
        | detections[("robust", 500)]
    )

    strict_detected = (
        detections[("strict", 250)]
        | detections[("strict", 500)]
    )

    any_support = (
        robust_support
        | strict_support
    )

    context = np.full(
        N,
        "HISTORICAL_CONTEXT_UNAVAILABLE",
        dtype=object,
    )

    context[any_support] = (
        "NO_CONVERSION_DETECTED_IN_AVAILABLE_CONTEXTS"
    )

    context[
        robust_detected
        & ~strict_detected
        & strict_support
    ] = "ROBUST_ONLY_DETECTION"

    context[
        robust_detected
        & ~strict_support
    ] = "ROBUST_DETECTION_STRICT_CONTEXT_UNAVAILABLE"

    context[
        strict_detected
        & ~robust_detected
        & robust_support
    ] = "STRICT_ONLY_DETECTION_REVIEW"

    context[
        strict_detected
        & ~robust_support
    ] = "STRICT_DETECTION_ROBUST_CONTEXT_UNAVAILABLE"

    context[
        robust_detected
        & strict_detected
    ] = "ROBUST_AND_STRICT_DETECTION"

    output[
        "historical_robust_detected_any_radius"
    ] = supported_boolean(
        robust_detected,
        robust_support,
    )

    output[
        "historical_strict_detected_any_radius"
    ] = supported_boolean(
        strict_detected,
        strict_support,
    )

    output[
        "historical_evidence_context_type"
    ] = context

    output[
        "historical_evidence_available"
    ] = any_support

    # Historical vigilance is supportive context.
    # It must not adjust DCI or frozen CP.

    pressure_and_history = supported_boolean(
        domain
        & (flagged == 1)
        & robust_detected,
        domain & robust_support,
    )

    output[
        "flagged_pressure_with_robust_historical_detection"
    ] = pressure_and_history

    output[
        "original_dual_concern_screen"
    ] = dual

    output[
        "original_managed_growth_screen"
    ] = managed

    output[
        "historical_support_role"
    ] = "SUPPORTING_CONTEXT_ONLY_NO_SCORE_MULTIPLICATION"

    output[
        "historical_evidence_is_proof_of_illegal_development"
    ] = False

    output[
        "historical_evidence_is_future_prediction"
    ] = False

    require(
        len(output) == N
        and output.grid_id.is_unique,
        "9,240 historical-context output rows",
    )

    require(
        output.loc[
            ~domain,
            "flagged_pressure_with_robust_historical_detection",
        ].isna().all(),
        "No flagged-pressure historical assertions "
        "outside pressure domain",
    )

    require(
        int(
            (
                output.historical_evidence_context_type
                == "HISTORICAL_CONTEXT_UNAVAILABLE"
            ).sum()
        )
        == int((~any_support).sum()),
        "Missing historical context explicitly preserved",
    )

    print("\nHistorical context categories:")

    print(
        output.historical_evidence_context_type
        .value_counts()
        .to_string()
    )

    # =====================================================
    # 5. EXPORT RESULTS
    # =====================================================

    print("\n[5] EXPORT CSV AND VERIFY")

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORT.mkdir(
        parents=True,
        exist_ok=True,
    )

    with TemporaryDirectory(
        prefix="18e_build_",
        dir=OUT_DIR,
    ) as temp:

        tmp_csv = (
            Path(temp) / OUT_CSV.name
        )

        output.to_csv(
            tmp_csv,
            index=False,
        )

        saved = pd.read_csv(
            tmp_csv
        )

        validate_grid(
            saved,
            "Exported historical-context CSV",
        )

        require(
            saved.historical_evidence_context_type.equals(
                output.historical_evidence_context_type
            ),
            "Historical categories preserved",
        )

        for mode in ("robust", "strict"):

            for radius in (250, 500):

                field = (
                    f"historical_g2b_{mode}_"
                    f"exposure_share_{radius}m"
                )

                require(
                    np.allclose(
                        number(saved, field),
                        number(output, field),
                        equal_nan=True,
                        rtol=0,
                        atol=1e-12,
                    ),
                    f"Export preserves {mode} "
                    f"{radius}m historical exposure",
                )

        require(
            not OUT_CSV.exists()
            and not OUT_QA.exists(),
            "No existing outputs overwritten",
        )

        tmp_csv.replace(
            OUT_CSV
        )

    # =====================================================
    # 6. VERIFY SOURCES
    # =====================================================

    print("\n[6] VERIFY INPUT FILE INTEGRITY")

    for p in inputs:

        require(
            sha256(p)
            == hashes[
                p.relative_to(ROOT).as_posix()
            ],
            f"Source unchanged: {p.name}",
        )

    # =====================================================
    # 7. FINAL QA
    # =====================================================

    qa = {

        "section":
            "05_planning_conflict_intervention",

        "stage":
            "18e_historical_green_loss_vigilance",

        "status": (
            "PASS_HISTORICAL_VIGILANCE_"
            "SUPPORT_ONLY_WITH_MISSINGNESS"
        ),

        "total_grid_cells":
            N,

        "pressure_model_domain_cells":
            int(domain.sum()),

        "complete_planning_typology_cells":
            int(full_type.sum()),

        "historical_context_available_cells":
            int(any_support.sum()),

        "historical_context_unavailable_cells":
            int((~any_support).sum()),

        "robust_detection_cells_any_radius":
            int(robust_detected.sum()),

        "strict_detection_cells_any_radius":
            int(strict_detected.sum()),

        "pressure_flagged_and_robust_historical_detection_cells":
            int(
                (
                    domain
                    & (flagged == 1)
                    & robust_detected
                ).sum()
            ),

        "support_counts": {
            f"{mode}_{radius}m": {
                "supported": int(
                    supports[(mode, radius)].sum()
                ),
                "detected": int(
                    detections[(mode, radius)].sum()
                ),
            }
            for mode in ("robust", "strict")
            for radius in (250, 500)
        },

        "evidence_categories": {
            str(k): int(v)
            for k, v in (
                output.historical_evidence_context_type
                .value_counts()
                .items()
            )
        },

        "historical_exposure_role":
            "SUPPORTING_CONTEXT_ONLY",

        "dci_or_cp_recalculated":
            False,

        "intervention_priority_assigned":
            False,

        "hypothetical_policy_thresholds_assigned":
            False,

        "limitations": [
            (
                "Nearby historical green-to-built "
                "conversion does not prove illegal "
                "development or predict future conversion."
            ),
            (
                "250m and 500m buffers overlap, "
                "so detections are not independent events."
            ),
            (
                "Robust and strict are alternative "
                "historical evidence definitions, "
                "not probabilities."
            ),
            (
                "Missing historical context is kept "
                "unavailable, not filled with zero."
            ),
            (
                "Out-of-domain pressure remains "
                "unavailable."
            ),
            (
                "Historical evidence is supporting "
                "context only and is not multiplied "
                "into DCI or Conservation Priority."
            ),
        ],

        "input_sha256":
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

    # =====================================================
    # 8. TERMINAL SUMMARY
    # =====================================================

    print("\n" + "=" * 78)

    print(qa["status"])

    print(
        "Historical-context available:",
        int(any_support.sum()),
    )

    print(
        "Historical-context unavailable:",
        int((~any_support).sum()),
    )

    print(
        "Robust historical detections:",
        int(robust_detected.sum()),
    )

    print(
        "Strict historical detections:",
        int(strict_detected.sum()),
    )

    print(
        "Frozen DCI / Conservation Priority changed: NO"
    )

    print(
        "Intervention classes assigned: NO"
    )

    print("\nCSV:", OUT_CSV)
    print("QA:", OUT_QA)

    print("=" * 78)


if __name__ == "__main__":
    main()
