
"""
J26-DS-338 | Section 05 | Step 18G

SEPARATE, EVIDENCE-AWARE PLANNING REVIEW QUEUES

No new expert weights.
No combined DCI + CCI index.
No legal priority classification.

Ranks are relative, reproducible queue positions.

Conservation Priority already contains pressure.
Never recompute or multiply it by pressure again.
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
    ROOT / "data/processed/planning_conflict_intervention"
)

REPORT = (
    ROOT / "outputs/reports/planning_conflict_intervention"
)

FILES = {
    "18c": (
        BASE / "development_conflict"
        / "18c_development_conflict_index_100m.csv"
    ),
    "18d": (
        BASE / "conflict_typology"
        / "18d_planning_conflict_typology_100m.csv"
    ),
    "18f": (
        BASE / "planning_intervention_rules"
        / "18f_explainable_planning_interventions_100m.csv"
    ),
}

QAS = {
    "18c": (
        REPORT / "18c_development_conflict_index_qa.json"
    ),
    "18d": (
        REPORT / "18d_planning_conflict_typology_qa.json"
    ),
    "18f": (
        REPORT / "18f_explainable_planning_interventions_qa.json"
    ),
}

EXPECTED_STATUSES = {
    "18c": (
        "PASS_DCI_VALID_PRESSURE_DOMAIN_"
        "CCI_FROZEN_SEPARATE"
    ),
    "18d": (
        "PASS_EXPLORATORY_PLANNING_TYPOLOGY_"
        "WITH_DOMAIN_LIMITS"
    ),
    "18f": (
        "PASS_EXPLAINABLE_PLANNING_REVIEW_RULES_"
        "PRIORITIZATION_PENDING"
    ),
}

OUT_DIR = BASE / "intervention_prioritization"

OUT_CSV = (
    OUT_DIR
    / "18g_separate_planning_review_queues_100m.csv"
)

OUT_QA = (
    REPORT
    / "18g_separate_planning_review_queues_qa.json"
)


# ============================================================
# 2. VERIFIED RESEARCH COUNTS
# ============================================================

TOTAL = 9240
PRESSURE_VALID = 4620
CP_VALID = 4609
PRESSURE_FLAGGED = 255
CP_UPPER = 1153


# ============================================================
# 3. HELPERS
# ============================================================

def require(condition, description):

    good = bool(condition)

    print(
        f"[{'PASS' if good else 'FAIL'}] "
        f"{description}"
    )

    if not good:
        raise ValueError(
            f"18G QA failed: {description}"
        )


def sha256(path):

    h = hashlib.sha256()

    with path.open("rb") as stream:

        for data in iter(
            lambda: stream.read(1024 * 1024),
            b"",
        ):
            h.update(data)

    return h.hexdigest()


def numeric(df, column):

    if column not in df.columns:
        raise KeyError(
            f"Required field absent: {column}"
        )

    return pd.to_numeric(
        df[column],
        errors="coerce",
    ).to_numpy(dtype=float)


def check_grid(df, label):

    require(
        len(df) == TOTAL
        and "grid_id" in df.columns
        and df["grid_id"].notna().all()
        and df["grid_id"].is_unique,
        f"{label}: 9,240 unique grid IDs",
    )


def rank_subset(values, ids, selected, name):

    """
    Rank 1 = highest score.

    For exact ties:
        ascending grid_id.

    Cells outside the selected queue:
        rank is missing (NaN).
    """

    indices = np.flatnonzero(selected)

    require(
        len(indices) > 0,
        f"{name}: nonempty queue",
    )

    require(
        np.isfinite(values[indices]).all(),
        f"{name}: finite ranking scores",
    )

    order = np.lexsort(
        (
            ids[indices],
            -values[indices],
        )
    )

    ranks = np.full(
        len(values),
        np.nan,
        dtype=float,
    )

    ranks[indices[order]] = np.arange(
        1,
        len(indices) + 1,
    )

    require(
        np.array_equal(
            np.sort(
                ranks[selected].astype(int)
            ),
            np.arange(
                1,
                len(indices) + 1,
            ),
        )
        and np.isnan(
            ranks[~selected]
        ).all(),
        (
            f"{name}: complete unique "
            f"1..{len(indices):,} ranks; "
            "others missing"
        ),
    )

    return ranks


# ============================================================
# 4. MAIN
# ============================================================

def main():

    print("=" * 78)
    print(
        "SECTION 05 | STEP 18G - "
        "SEPARATE PLANNING REVIEW QUEUES"
    )
    print("=" * 78)

    inputs = [
        *FILES.values(),
        *QAS.values(),
    ]

    for path in inputs:
        if not path.is_file():
            raise FileNotFoundError(path)

    for path in (
        OUT_CSV,
        OUT_QA,
    ):
        if path.exists():
            raise FileExistsError(
                "Existing output must not be "
                f"overwritten: {path}"
            )

    input_hashes = {
        p.relative_to(ROOT).as_posix(): sha256(p)
        for p in inputs
    }

    # ========================================================
    # STEP 1 - VERIFY PREVIOUS RESEARCH
    # ========================================================

    print(
        "\n[1] VERIFY RESEARCH RELEASES "
        "AND PROVENANCE"
    )

    qa = {
        name: json.loads(
            path.read_text(encoding="utf-8")
        )
        for name, path in QAS.items()
    }

    for name in FILES:

        require(
            qa[name].get("status")
            == EXPECTED_STATUSES[name],
            (
                f"{name.upper()}: approved "
                "preceding stage status"
            ),
        )

        if name == "18c":

            documented_hash = (
                qa[name]["outputs"]["csv_sha256"]
            )

        else:

            documented_hash = (
                qa[name]["output"]["sha256"]
            )

        require(
            documented_hash == sha256(FILES[name]),
            (
                f"{name.upper()}: exported "
                "CSV hash verified"
            ),
        )

    for name in (
        "18c",
        "18d",
    ):

        relative = (
            FILES[name]
            .relative_to(ROOT)
            .as_posix()
        )

        require(
            qa["18f"]["input_sha256"].get(
                relative
            ) == sha256(FILES[name]),
            (
                f"18F uses exact "
                f"{name.upper()} input"
            ),
        )

    require(
        qa["18c"].get(
            "pressure_is_calibrated_probability"
        ) is False,
        (
            "Future urbanization pressure is "
            "not a calibrated probability"
        ),
    )

    require(
        qa["18f"].get(
            "dci_and_cci_combined_into_new_risk_score"
        ) is False,
        "No preceding combined DCI/CCI score",
    )

    # ========================================================
    # STEP 2 - LOAD AND ALIGN DATA
    # ========================================================

    print(
        "\n[2] LOAD AND RECONCILE SOURCE DATA"
    )

    f = pd.read_csv(
        FILES["18f"]
    )

    c = pd.read_csv(
        FILES["18c"]
    )

    d = pd.read_csv(
        FILES["18d"]
    )

    for label, data in (
        ("18F", f),
        ("18C", c),
        ("18D", d),
    ):

        check_grid(
            data,
            label,
        )

    require(
        set(f.grid_id)
        == set(c.grid_id)
        == set(d.grid_id),
        "All grid identities match",
    )

    c = (
        c.set_index("grid_id")
        .loc[f.grid_id]
        .reset_index()
    )

    d = (
        d.set_index("grid_id")
        .loc[f.grid_id]
        .reset_index()
    )

    ids = (
        f.grid_id
        .astype(str)
        .to_numpy()
    )

    p = numeric(
        f,
        "urbanization_pressure_score",
    )

    s = numeric(
        f,
        "development_suitability_score",
    )

    dci = numeric(
        f,
        "development_conflict_index_0_1",
    )

    cp = numeric(
        f,
        "frozen_conservation_conflict_index_0_1",
    )

    flag = numeric(
        f,
        "pressure_threshold_flag_frozen",
    )

    area = numeric(
        f,
        "kmc_overlap_area_m2",
    )

    route = (
        f.review_rule_id
        .astype(str)
        .to_numpy()
    )

    domain = np.isfinite(p)

    cp_domain = np.isfinite(cp)

    flagged = (
        domain
        & (flag == 1)
    )

    upper_cp = (
        d.relative_frozen_cp_screen
        .eq(
            "HIGHEST_25_PERCENT_COMMON_DOMAIN"
        )
        .to_numpy(dtype=bool)
    )

    joint = (
        route == "R02_JOINT_CONCERN"
    )

    gap = (
        route == "R01_CONSERVATION_EVIDENCE_GAP"
    )

    managed = (
        route
        == "R05_MANAGED_GROWTH_FEASIBILITY"
    )

    # ========================================================
    # STEP 3 - VERIFY INPUTS AND DOMAINS
    # ========================================================

    print(
        "\n[3] VALIDATE DOMAINS, "
        "INDICES AND REVIEW RULES"
    )

    require(
        int(domain.sum()) == PRESSURE_VALID,
        "4,620 pressure-domain cells",
    )

    require(
        int(cp_domain.sum()) == CP_VALID
        and np.all(
            ~cp_domain | domain
        ),
        (
            "4,609 frozen CP cells confined "
            "to pressure domain"
        ),
    )

    require(
        np.array_equal(
            np.isfinite(dci),
            domain,
        ),
        "DCI available exactly in pressure domain",
    )

    require(
        np.isfinite(s).all()
        and np.all(
            (s >= 0)
            & (s <= 1)
        ),
        (
            "Final AHP suitability "
            "complete and bounded"
        ),
    )

    require(
        np.allclose(
            dci[domain],
            p[domain] * (1 - s[domain]),
            atol=1e-12,
            rtol=0,
        ),
        "DCI formula independently reproduced",
    )

    require(
        np.allclose(
            cp,
            numeric(
                c,
                "conservation_conflict_index_0_1",
            ),
            equal_nan=True,
            atol=1e-12,
            rtol=0,
        ),
        "CCI is identical to frozen CP",
    )

    require(
        np.allclose(
            dci,
            numeric(
                c,
                "development_conflict_index_0_1",
            ),
            equal_nan=True,
            atol=1e-12,
            rtol=0,
        ),
        "18F DCI matches 18C DCI",
    )

    require(
        np.allclose(
            p,
            numeric(
                c,
                "urbanization_pressure_score",
            ),
            equal_nan=True,
            atol=1e-12,
            rtol=0,
        ),
        "18F pressure matches 18C pressure",
    )

    require(
        np.allclose(
            s,
            numeric(
                c,
                "development_suitability_score",
            ),
            atol=1e-12,
            rtol=0,
        ),
        "18F suitability matches 18C suitability",
    )

    require(
        np.isnan(
            flag[~domain]
        ).all()
        and int(
            flagged.sum()
        ) == PRESSURE_FLAGGED,
        (
            "255 flagged cells; "
            "out-of-domain flags missing"
        ),
    )

    require(
        int(
            upper_cp.sum()
        ) == CP_UPPER
        and np.all(
            ~upper_cp | cp_domain
        ),
        (
            "1,153 frozen CP upper-quartile cells"
        ),
    )

    require(
        int(joint.sum()) == 7
        and int(gap.sum()) == 11
        and int(managed.sum()) == 1,
        (
            "7 joint reviews, 11 evidence gaps, "
            "1 feasibility screen"
        ),
    )

    require(
        np.array_equal(
            route == "R00_OUTSIDE_PRESSURE_DOMAIN",
            ~domain,
        ),
        (
            "Outside-domain grids have no "
            "planning-conflict inference"
        ),
    )

    require(
        np.array_equal(
            gap,
            domain & ~cp_domain,
        ),
        (
            "Evidence-gap route matches "
            "missing conservation inputs"
        ),
    )

    lower_suitability = (
        d.relative_suitability_screen
        .eq(
            "LOWEST_25_PERCENT_DOMAIN_RELATIVE"
        )
        .to_numpy(dtype=bool)
    )

    require(
        np.array_equal(
            joint,
            flagged
            & upper_cp
            & lower_suitability,
        ),
        (
            "Seven joint-concern flags "
            "independently reproduced"
        ),
    )

    require(
        np.all(
            ~managed | flagged
        )
        and np.all(
            ~managed | cp_domain
        ),
        (
            "Feasibility screens use flagged, "
            "full-evidence grids"
        ),
    )

    conservation_routes = np.isin(
        route,
        [
            "R02_JOINT_CONCERN",
            "R03_CONSERVATION_SAFEGUARD",
            "R07_CONSERVATION_CONTEXT",
        ],
    )

    require(
        np.array_equal(
            upper_cp,
            conservation_routes,
        ),
        (
            "Conservation review queue "
            "agrees with original rulebook"
        ),
    )

    counts = (
        f.review_rule_id
        .value_counts()
        .to_dict()
    )

    require(
        all(
            counts.get(key, 0) == int(value)
            for key, value in (
                qa["18f"]["review_rule_counts"].items()
            )
        ),
        (
            "Review-stream counts "
            "agree with 18F QA"
        ),
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
        "Clipped KMC grid areas verified",
    )

    # ========================================================
    # STEP 4 - CALCULATE SEPARATE RANKS
    # ========================================================

    print(
        "\n[4] COMPUTE SEPARATE RANKING DIMENSIONS"
    )

    ranks = {

        "dci_rank_in_pressure_domain":
            rank_subset(
                dci,
                ids,
                domain,
                "All valid DCI grids",
            ),

        "cp_rank_in_conservation_domain":
            rank_subset(
                cp,
                ids,
                cp_domain,
                "All frozen CP grids",
            ),

        "flagged_development_review_rank_by_dci":
            rank_subset(
                dci,
                ids,
                flagged,
                "Frozen pressure-flagged review queue",
            ),

        "upper_cp_conservation_review_rank_by_cp":
            rank_subset(
                cp,
                ids,
                upper_cp,
                "Upper CP-quartile review queue",
            ),

        "joint_review_rank_by_dci":
            rank_subset(
                dci,
                ids,
                joint,
                "Joint-concern DCI ranking",
            ),

        "joint_review_rank_by_cp":
            rank_subset(
                cp,
                ids,
                joint,
                "Joint-concern CP ranking",
            ),

        "missing_conservation_evidence_rank_by_dci":
            rank_subset(
                dci,
                ids,
                gap,
                "Evidence-gap DCI context ranking",
            ),

        "managed_growth_screen_rank_by_suitability":
            rank_subset(
                s,
                ids,
                managed,
                "Managed-growth suitability screen",
            ),
    }

    # No DCI + CP sum, multiplication, or
    # weighted combined ranking is calculated.

    # ========================================================
    # STEP 5 - ASSEMBLE FINAL RANKING TABLE
    # ========================================================

    print(
        "\n[5] CREATE REVIEW-QUEUE OUTPUT"
    )

    result = pd.DataFrame({

        "grid_id":
            f.grid_id,

        "kmc_overlap_area_m2":
            area,

        "pressure_origin_year":
            2025,

        "pressure_period":
            "2026-2028",

        "pressure_domain_status":
            f.pressure_domain_status,

        "review_rule_id":
            f.review_rule_id,

        "review_stream":
            f.review_stream,

        "review_action":
            f.review_action,

        "explanation_of_rule_trigger":
            f.explanation_of_rule_trigger,

        "urbanization_pressure_score":
            p,

        "pressure_threshold_flag_frozen":
            pd.Series(flag).astype("Int64"),

        "development_suitability_score":
            s,

        "development_conflict_index_0_1":
            dci,

        "frozen_conservation_priority_0_1":
            cp,

        "historical_context_type":
            f.historical_context_type,

        "historical_robust_exposure_share_500m":
            numeric(
                f,
                "historical_robust_exposure_share_500m",
            ),

        "in_flagged_development_review_queue":
            pd.Series(flagged)
            .where(domain)
            .astype("boolean"),

        "in_upper_cp_conservation_review_queue":
            pd.Series(upper_cp)
            .where(cp_domain)
            .astype("boolean"),

        "in_joint_concern_review":
            pd.Series(joint)
            .where(cp_domain)
            .astype("boolean"),

        "missing_conservation_evidence":
            gap,

        "managed_growth_feasibility_screen":
            pd.Series(managed)
            .where(cp_domain)
            .astype("boolean"),

        "official_priority_or_development_approval":
            False,

        "ranking_method": (
            "SEPARATE_INDICES_DESCENDING_"
            "WITH_GRID_ID_TIEBREAK"
        ),
    })

    for column, values in ranks.items():

        result[column] = (
            pd.Series(values)
            .astype("Int64")
        )

    check_grid(
        result,
        "18G separate review queues",
    )

    require(
        "composite_priority_score"
        not in result.columns,
        "No arbitrary combined priority index",
    )

    require(
        result.loc[
            ~domain,
            "dci_rank_in_pressure_domain",
        ].isna().all(),
        "Out-of-domain DCI ranks remain missing",
    )

    require(
        result.loc[
            ~cp_domain,
            "cp_rank_in_conservation_domain",
        ].isna().all(),
        "Missing CP scores are never ranked",
    )

    # ========================================================
    # STEP 6 - EXPORT AND VERIFY
    # ========================================================

    print(
        "\n[6] EXPORT AND VERIFY"
    )

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORT.mkdir(
        parents=True,
        exist_ok=True,
    )

    with TemporaryDirectory(
        prefix="18g_build_",
        dir=OUT_DIR,
    ) as folder:

        temporary = (
            Path(folder) / OUT_CSV.name
        )

        result.to_csv(
            temporary,
            index=False,
        )

        saved = pd.read_csv(
            temporary
        )

        check_grid(
            saved,
            "Exported 18G CSV",
        )

        require(
            saved.review_rule_id.equals(
                result.review_rule_id
            ),
            "Review routes preserved during export",
        )

        for name, values in ranks.items():

            require(
                np.allclose(
                    numeric(saved, name),
                    values,
                    atol=0,
                    rtol=0,
                    equal_nan=True,
                ),
                (
                    "Exported ranks preserved: "
                    f"{name}"
                ),
            )

        for field, expected in [

            (
                "urbanization_pressure_score",
                p,
            ),

            (
                "development_suitability_score",
                s,
            ),

            (
                "development_conflict_index_0_1",
                dci,
            ),

            (
                "frozen_conservation_priority_0_1",
                cp,
            ),

        ]:

            require(
                np.allclose(
                    numeric(saved, field),
                    expected,
                    atol=1e-12,
                    rtol=0,
                    equal_nan=True,
                ),
                (
                    f"Export preserves {field} "
                    "and missingness"
                ),
            )

        require(
            not OUT_CSV.exists()
            and not OUT_QA.exists(),
            "Existing outputs not overwritten",
        )

        temporary.replace(
            OUT_CSV
        )

    # ========================================================
    # STEP 7 - VERIFY ORIGINAL FILES
    # ========================================================

    print(
        "\n[7] VERIFY SOURCE INTEGRITY"
    )

    for path in inputs:

        relative = (
            path.relative_to(ROOT)
            .as_posix()
        )

        require(
            sha256(path)
            == input_hashes[relative],
            f"Input unchanged: {path.name}",
        )

    # ========================================================
    # STEP 8 - RESEARCH QA REPORT
    # ========================================================

    print(
        "\n[8] SAVE REPRODUCIBLE QUEUE REPORT"
    )

    summary = {

        "section":
            "05_planning_conflict_intervention",

        "stage":
            "18g_separate_planning_review_queues",

        "status": (
            "PASS_SEPARATE_REVIEW_QUEUE_RANKINGS_"
            "SENSITIVITY_PENDING"
        ),

        "total_grid_cells":
            TOTAL,

        "pressure_domain_cells":
            int(domain.sum()),

        "conservation_score_cells":
            int(cp_domain.sum()),

        "pressure_flagged_review_queue":
            int(flagged.sum()),

        "upper_cp_conservation_review_queue":
            int(upper_cp.sum()),

        "joint_concern_review_queue":
            int(joint.sum()),

        "evidence_gap_cells":
            int(gap.sum()),

        "managed_growth_feasibility_screen_cells":
            int(managed.sum()),

        "ranked_domains": {
            name: int(
                np.isfinite(values).sum()
            )
            for name, values in ranks.items()
        },

        "tie_break": (
            "Descending score then ascending grid_id"
        ),

        "weighting_or_aggregation": (
            "NO_COMPOSITE_INDEX_OR_NEW_WEIGHTS"
        ),

        "frozen_cp_already_contains_pressure":
            True,

        "historical_context_used_for_ranking":
            False,

        "expert_ahp_weights_modified":
            False,

        "source_outputs_modified":
            False,

        "official_priorities_issued":
            False,

        "limitations": [

            (
                "Separate queues are exploratory "
                "review orderings, not official "
                "priority tiers."
            ),

            (
                "CP already includes pressure; "
                "DCI and CP must not be summed "
                "or multiplied."
            ),

            (
                "Ranks from different queues "
                "are not directly comparable."
            ),

            (
                "Pressure is a relative model-derived "
                "score, not a calibrated probability."
            ),

            (
                "Out-of-domain and missing conservation "
                "scores are never imputed."
            ),

            (
                "Historical conversion is supporting "
                "context and does not affect ranking."
            ),

            (
                "The single managed-growth screening "
                "grid still requires official "
                "feasibility checks."
            ),

            (
                "Review-queue sensitivity and final "
                "GIS packaging remain pending."
            ),
        ],

        "input_sha256":
            input_hashes,

        "output": {

            "csv":
                OUT_CSV.relative_to(
                    ROOT
                ).as_posix(),

            "sha256":
                sha256(OUT_CSV),
        },
    }

    OUT_QA.write_text(
        json.dumps(
            summary,
            indent=2,
        ),
        encoding="utf-8",
    )

    # ========================================================
    # STEP 9 - TERMINAL SUMMARY
    # ========================================================

    print("\n" + "=" * 78)

    print(
        summary["status"]
    )

    print(
        "Valid DCI ranks:",
        int(domain.sum()),
    )

    print(
        "Valid CP ranks:",
        int(cp_domain.sum()),
    )

    print(
        "Flagged-pressure DCI review queue:",
        int(flagged.sum()),
    )

    print(
        "Upper-CP conservation review queue:",
        int(upper_cp.sum()),
    )

    print(
        "Joint-concern reviews:",
        int(joint.sum()),
    )

    print(
        "Evidence gaps:",
        int(gap.sum()),
    )

    print(
        "Managed-growth screens:",
        int(managed.sum()),
    )

    print(
        "Combined DCI/CP score created: NO"
    )

    print(
        "Official planning priorities issued: NO"
    )

    print("\nCSV:", OUT_CSV)
    print("QA:", OUT_QA)

    print("=" * 78)


if __name__ == "__main__":
    main()
