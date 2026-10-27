
"""
J26-DS-338 | Section 05 | Step 18C

DEVELOPMENT CONFLICT INDEX

DCI = P * (1 - S)

P = Frozen urbanization pressure score
S = Final expert-AHP development suitability

DCI is calculated only inside the pressure-model domain.

Conservation Conflict Indicator (CCI) is the existing
frozen Conservation Priority (CP), not P * CP.

No new classes, thresholds, models or predictions.
Frozen Sections 02, 03 and 04 remain unchanged.
"""

from pathlib import Path
from tempfile import TemporaryDirectory
import hashlib
import json

import numpy as np
import pandas as pd


# ============================================================
# 1. PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

BASE = (
    ROOT
    / "data/processed/planning_conflict_intervention"
)

REPORT = (
    ROOT
    / "outputs/reports/planning_conflict_intervention"
)

SOURCE_CSV = (
    BASE
    / "common_grid"
    / "18b_common_grid_integration_100m.csv"
)

SOURCE_QA = (
    REPORT
    / "18b_common_grid_integration_qa.json"
)

OUTPUT_DIR = (
    BASE / "development_conflict"
)

OUTPUT_CSV = (
    OUTPUT_DIR
    / "18c_development_conflict_index_100m.csv"
)

OUTPUT_QA = (
    REPORT
    / "18c_development_conflict_index_qa.json"
)


# ============================================================
# 2. FROZEN CONTRACT
# ============================================================

N = 9240
N_PRESSURE = 4620
N_CP = 4609

EXPECTED_AREA_KM2 = 88.82921581108494

TOL = 1e-12


# ============================================================
# 3. HELPER FUNCTIONS
# ============================================================

def require(condition, description):

    passed = bool(condition)

    print(
        f"[{'PASS' if passed else 'FAIL'}] "
        f"{description}"
    )

    if not passed:
        raise ValueError(
            f"Step 18C QA failed: {description}"
        )


def sha256(path):

    h = hashlib.sha256()

    with path.open("rb") as stream:
        for block in iter(
            lambda: stream.read(1024 * 1024),
            b"",
        ):
            h.update(block)

    return h.hexdigest()


def numeric(data, column):

    if column not in data.columns:
        raise KeyError(
            f"Required column missing: {column}"
        )

    return pd.to_numeric(
        data[column],
        errors="coerce",
    ).to_numpy(dtype=float)


def flags(data, column):

    if column not in data.columns:
        raise KeyError(
            f"Required column missing: {column}"
        )

    values = (
        data[column]
        .astype("string")
        .str.strip()
        .str.lower()
    )

    require(
        values.isin(
            ["true", "false", "1", "0"]
        ).all(),
        f"{column} is strictly boolean",
    )

    return values.isin(
        ["true", "1"]
    ).to_numpy(dtype=bool)


def summary(values):

    selected = np.asarray(
        values,
        dtype=float,
    )

    selected = selected[
        np.isfinite(selected)
    ]

    return {
        "count": int(len(selected)),
        "min": float(np.min(selected)),
        "p05": float(
            np.quantile(selected, 0.05)
        ),
        "median": float(
            np.median(selected)
        ),
        "mean": float(
            np.mean(selected)
        ),
        "p95": float(
            np.quantile(selected, 0.95)
        ),
        "max": float(np.max(selected)),
    }


# ============================================================
# 4. MAIN
# ============================================================

def main():

    print("=" * 78)
    print(
        "SECTION 05 | STEP 18C - "
        "DEVELOPMENT CONFLICT INDEX"
    )
    print("=" * 78)

    for path in (
        SOURCE_CSV,
        SOURCE_QA,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    for path in (
        OUTPUT_CSV,
        OUTPUT_QA,
    ):
        if path.exists():
            raise FileExistsError(
                "Refusing to overwrite existing file: "
                f"{path}"
            )

    input_hashes = {
        p.relative_to(ROOT).as_posix(): sha256(p)
        for p in (SOURCE_CSV, SOURCE_QA)
    }

    # ========================================================
    # STEP 1 - VERIFY COMMON GRID QA
    # ========================================================

    print(
        "\n[1] VERIFY SECTION 05 COMMON-GRID RELEASE"
    )

    qa18b = json.loads(
        SOURCE_QA.read_text(
            encoding="utf-8"
        )
    )

    require(
        qa18b.get("status")
        == "PASS_COMMON_GRID_FINAL_EXPERT_AHP_INTEGRATION",
        "18B integration QA passed",
    )

    require(
        qa18b.get("grid_cells") == N
        and qa18b.get("crs") == "EPSG:32644",
        "18B count and projection verified",
    )

    require(
        qa18b.get("outputs", {}).get(
            "csv_sha256"
        ) == sha256(SOURCE_CSV),
        "Input CSV matches frozen 18B QA hash",
    )

    require(
        qa18b.get("suitability_release")
        == "16g_01"
        and qa18b.get("suitability_criteria")
        == "C1_C2_C4",
        "Released expert-AHP suitability provenance",
    )

    require(
        qa18b.get(
            "pressure_not_a_calibrated_probability"
        ) is True,
        "Pressure has correct nonprobability interpretation",
    )

    recorded_counts = qa18b.get(
        "domain_counts", {}
    )

    require(
        recorded_counts.get("pressure_in_domain")
        == N_PRESSURE
        and recorded_counts.get(
            "dci_eligible_for_next_stage"
        ) == N_PRESSURE
        and recorded_counts.get(
            "frozen_conservation_priority_available"
        ) == N_CP,
        "18B domain contract verified",
    )

    # ========================================================
    # STEP 2 - VALIDATE INPUT DOMAINS
    # ========================================================

    print(
        "\n[2] VALIDATE JOINED GRID AND DOMAINS"
    )

    df = pd.read_csv(SOURCE_CSV)

    required = [
        "grid_id",
        "kmc_overlap_area_m2",
        "final_inference_ready",
        "pressure_domain_status_section05",
        "dci_input_status",
        "cci_input_status",
        "joint_evidence_status",
        "eligibility_origin_year",
        "eligibility_pressure_period",
        "urbanization_pressure_score",
        "development_suitability_score",
        "primary_conservation_priority_0_1",
        "green_importance_index_0_1",
        "suitability_criteria_set",
        "score_interpretation",
        "threshold_flag",
    ]

    require(
        set(required).issubset(df.columns),
        "All required 18B fields available",
    )

    require(
        len(df) == N
        and df.grid_id.notna().all()
        and df.grid_id.is_unique,
        "All 9,240 grid IDs are unique",
    )

    domain = flags(
        df, "final_inference_ready"
    )

    require(
        int(domain.sum()) == N_PRESSURE
        and int((~domain).sum())
        == N - N_PRESSURE,
        "4,620 scored and 4,620 out-of-domain cells",
    )

    require(
        np.array_equal(
            df[
                "pressure_domain_status_section05"
            ].eq("IN_MODEL_DOMAIN").to_numpy(),
            domain,
        ),
        "Domain labels agree with eligibility",
    )

    require(
        np.array_equal(
            df[
                "dci_input_status"
            ].eq("READY_FOR_18C_DCI").to_numpy(),
            domain,
        ),
        "18B DCI readiness labels agree",
    )

    origin = numeric(
        df, "eligibility_origin_year"
    )

    require(
        np.isfinite(origin).all()
        and np.all(origin == 2025)
        and df[
            "eligibility_pressure_period"
        ].eq("2026-2028").all(),
        "Frozen 2025-origin 2026-2028 horizon",
    )

    area = numeric(
        df, "kmc_overlap_area_m2"
    )

    require(
        np.isfinite(area).all()
        and np.all(area > 0)
        and np.isclose(
            area.sum() / 1e6,
            EXPECTED_AREA_KM2,
            atol=0.001,
            rtol=0,
        ),
        "Valid clipped areas and KMC total area",
    )

    pressure = numeric(
        df, "urbanization_pressure_score"
    )

    suitability = numeric(
        df, "development_suitability_score"
    )

    cp = numeric(
        df, "primary_conservation_priority_0_1"
    )

    gi = numeric(
        df, "green_importance_index_0_1"
    )

    for label, values in (
        ("Pressure", pressure),
        ("Suitability", suitability),
        ("Frozen CP", cp),
        ("Frozen GI", gi),
    ):

        valid = values[
            np.isfinite(values)
        ]

        require(
            not np.isinf(values).any()
            and np.all(valid >= -TOL)
            and np.all(valid <= 1 + TOL),
            f"{label}: values in [0,1] when present",
        )

    require(
        np.array_equal(
            np.isfinite(pressure),
            domain,
        ),
        "Pressure only exists in model domain",
    )

    require(
        np.isfinite(suitability).all(),
        "Suitability available for all grids",
    )

    require(
        np.count_nonzero(
            np.isfinite(cp)
        ) == N_CP
        and np.all(
            ~np.isfinite(cp) | domain
        ),
        "Frozen CP available only for 4,609 domain grids",
    )

    require(
        np.array_equal(
            df[
                "cci_input_status"
            ].eq("FROZEN_CP_AVAILABLE").to_numpy(),
            np.isfinite(cp),
        ),
        "CCI availability agrees with frozen CP",
    )

    require(
        df[
            "suitability_criteria_set"
        ].eq("C1_C2_C4").all(),
        "Final suitability uses C1/C2/C4",
    )

    require(
        df.loc[
            domain, "score_interpretation"
        ].eq(
            "MODEL_DERIVED_PRESSURE_SCORE_"
            "NOT_CALIBRATED_PROBABILITY"
        ).all(),
        "Pressure is not a calibrated probability",
    )

    threshold_flags = numeric(
        df, "threshold_flag"
    )

    require(
        np.isnan(
            threshold_flags[~domain]
        ).all()
        and int(
            np.count_nonzero(
                threshold_flags[domain] == 1
            )
        ) == 255,
        "Frozen pressure threshold flags preserved",
    )

    # ========================================================
    # STEP 3 - CALCULATE DCI
    # ========================================================

    print(
        "\n[3] CALCULATE DEVELOPMENT CONFLICT INDEX"
    )

    dci = np.full(
        N,
        np.nan,
        dtype=float,
    )

    dci[domain] = (
        pressure[domain]
        * (1.0 - suitability[domain])
    )

    require(
        np.array_equal(
            np.isfinite(dci),
            domain,
        ),
        "DCI exists exactly for 4,620 in-domain grids",
    )

    require(
        np.isnan(dci[~domain]).all(),
        "Out-of-domain DCI remains missing",
    )

    require(
        np.all(dci[domain] >= -TOL)
        and np.all(dci[domain] <= 1 + TOL),
        "Valid DCI within [0,1]",
    )

    require(
        np.all(
            dci[domain]
            <= pressure[domain] + TOL
        ),
        "DCI does not exceed pressure",
    )

    independent_dci = (
        pressure[domain]
        - pressure[domain]
        * suitability[domain]
    )

    require(
        np.allclose(
            dci[domain],
            independent_dci,
            atol=1e-12,
            rtol=0,
        ),
        "DCI independently reproduced as P - P*S",
    )

    # ========================================================
    # STEP 4 - PRESERVE FROZEN CCI
    # ========================================================

    print(
        "\n[4] PRESERVE SEPARATE CONSERVATION CONFLICT"
    )

    # CCI is exactly the already-frozen
    # Conservation Priority (CP).
    #
    # Do not multiply by pressure again.

    cci = cp.copy()

    require(
        np.allclose(
            cci,
            cp,
            atol=0,
            rtol=0,
            equal_nan=True,
        ),
        "CCI equals frozen CP without recalculation",
    )

    require(
        int(
            np.count_nonzero(
                domain & ~np.isfinite(cci)
            )
        ) == 11,
        "11 DCI-valid grids have unavailable CCI",
    )

    require(
        int(
            np.count_nonzero(
                ~domain & ~np.isfinite(cci)
            )
        ) == N - N_PRESSURE,
        "No CCI values outside pressure domain",
    )

    # ========================================================
    # STEP 5 - ASSEMBLE OUTPUT
    # ========================================================

    print(
        "\n[5] ASSEMBLE AUDITABLE DCI OUTPUT"
    )

    result = pd.DataFrame({

        "grid_id":
            df["grid_id"],

        "kmc_overlap_area_m2":
            area,

        "origin_year":
            origin.astype(int),

        "pressure_period":
            df["eligibility_pressure_period"],

        "pressure_domain_status":
            df["pressure_domain_status_section05"],

        "urbanization_pressure_score":
            pressure,

        "development_suitability_score":
            suitability,

        "suitability_shortfall_0_1":
            1.0 - suitability,

        "development_conflict_index_0_1":
            dci,

        "development_conflict_status":
            np.where(
                domain,
                "VALID_PRESSURE_SUITABILITY_DOMAIN",
                "NOT_IN_PRESSURE_MODEL_DOMAIN",
            ),

        "conservation_conflict_index_0_1":
            cci,

        "conservation_conflict_status":
            df["cci_input_status"],

        "joint_evidence_status":
            df["joint_evidence_status"],

        "pressure_threshold_flag_frozen":
            threshold_flags,
    })

    require(
        len(result) == N
        and result.grid_id.is_unique,
        "9,240 unique output grids",
    )

    require(
        "development_conflict_class"
        not in result.columns,
        "No unapproved conflict classification",
    )

    # ========================================================
    # STEP 6 - EXPORT OUTPUT
    # ========================================================

    print("\n[6] EXPORT AND VERIFY")

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORT.mkdir(
        parents=True,
        exist_ok=True,
    )

    with TemporaryDirectory(
        prefix="18c_build_",
        dir=OUTPUT_DIR,
    ) as folder:

        temp_csv = (
            Path(folder) / OUTPUT_CSV.name
        )

        result.to_csv(
            temp_csv,
            index=False,
        )

        saved = pd.read_csv(
            temp_csv
        )

        require(
            len(saved) == N
            and saved.grid_id.is_unique,
            "Exported CSV preserves all grid IDs",
        )

        for column, expected in (
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
        ):

            actual = numeric(
                saved,
                column,
            )

            require(
                np.allclose(
                    actual,
                    expected,
                    atol=1e-12,
                    rtol=0,
                    equal_nan=True,
                ),
                f"Export preserves values: {column}",
            )

        if (
            OUTPUT_CSV.exists()
            or OUTPUT_QA.exists()
        ):
            raise FileExistsError(
                "Output appeared during processing"
            )

        temp_csv.replace(
            OUTPUT_CSV
        )

    # ========================================================
    # STEP 7 - VERIFY INPUT INTEGRITY
    # ========================================================

    print(
        "\n[7] VERIFY INPUT INTEGRITY"
    )

    for path in (
        SOURCE_CSV,
        SOURCE_QA,
    ):

        key = (
            path.relative_to(ROOT).as_posix()
        )

        require(
            sha256(path)
            == input_hashes[key],
            f"Source unchanged: {path.name}",
        )

    # ========================================================
    # STEP 8 - CREATE QA REPORT
    # ========================================================

    print(
        "\n[8] GENERATE DEVELOPMENT CONFLICT QA"
    )

    dci_distribution = summary(
        dci[domain]
    )

    qa = {

        "section":
            "05_planning_conflict_intervention",

        "stage":
            "18c_development_conflict_index",

        "status": (
            "PASS_DCI_VALID_PRESSURE_DOMAIN_"
            "CCI_FROZEN_SEPARATE"
        ),

        "method":
            "DCI=P*(1-S); CCI=FROZEN_CP",

        "study_area_grid_cells":
            N,

        "pressure_origin_year":
            2025,

        "pressure_period":
            "2026-2028",

        "valid_dci_cells":
            int(np.isfinite(dci).sum()),

        "dci_unavailable_pressure_not_in_domain":
            int(np.isnan(dci).sum()),

        "frozen_cci_available_cells":
            int(np.isfinite(cci).sum()),

        "dci_without_cci_cells":
            int(
                np.count_nonzero(
                    domain & ~np.isfinite(cci)
                )
            ),

        "suitability_complete_cells":
            int(
                np.isfinite(suitability).sum()
            ),

        "dci_distribution_in_model_domain":
            dci_distribution,

        "dci_area_weighted_mean_in_model_domain":
            float(
                np.average(
                    dci[domain],
                    weights=area[domain],
                )
            ),

        "dci_domain_area_km2":
            float(
                area[domain].sum() / 1e6
            ),

        "cci_is_frozen_conservation_priority":
            True,

        "additional_pressure_multiplication_of_cci":
            False,

        "suitability_criteria":
            "C1_C2_C4",

        "pressure_is_calibrated_probability":
            False,

        "new_dci_classification_or_threshold":
            False,

        "source_files_modified":
            False,

        "limitations": [
            (
                "DCI is only defined where frozen "
                "pressure-model scores exist."
            ),
            (
                "Pressure scores are model-derived "
                "relative scores, not calibrated probabilities."
            ),
            (
                "Suitability covers road proximity, "
                "road density and slope, not comprehensive "
                "legal or flood-hazard suitability."
            ),
            (
                "DCI indicates potential pressure-suitability "
                "tension, not proven illegal development."
            ),
            (
                "CCI is unchanged Section 04 Conservation "
                "Priority, not another predicted risk."
            ),
        ],

        "input_sha256":
            input_hashes,

        "outputs": {
            "csv":
                OUTPUT_CSV.relative_to(
                    ROOT
                ).as_posix(),

            "csv_sha256":
                sha256(OUTPUT_CSV),
        },
    }

    OUTPUT_QA.write_text(
        json.dumps(
            qa,
            indent=2,
        ),
        encoding="utf-8",
    )

    # ========================================================
    # STEP 9 - TERMINAL SUMMARY
    # ========================================================

    print("\n" + "=" * 78)

    print(qa["status"])

    print(
        "DCI valid:",
        int(np.isfinite(dci).sum()),
    )

    print(
        "DCI unavailable:",
        int(np.isnan(dci).sum()),
    )

    print(
        "Frozen CCI valid:",
        int(np.isfinite(cci).sum()),
    )

    print(
        "DCI mean (valid domain):",
        f"{dci_distribution['mean']:.6f}",
    )

    print(
        "DCI median (valid domain):",
        f"{dci_distribution['median']:.6f}",
    )

    print(
        "Out-of-domain values preserved: YES"
    )

    print(
        "Expert suitability modified: NO"
    )

    print(
        "Frozen Conservation Priority modified: NO"
    )

    print(
        "Conflict classes assigned: NO"
    )

    print("\nCSV:", OUTPUT_CSV)
    print("QA:", OUTPUT_QA)

    print("=" * 78)


if __name__ == "__main__":
    main()
