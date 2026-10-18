
"""
J26-DS-338 | Section 05 | Step 5.10 (18J)
Final technical QA and consolidated GIS packaging.

Source models and scores are NEVER recalculated.
Hotspot groups and review categories remain EXPLORATORY and cutoff-sensitive.
GN polygon coverage gaps and missing model-domain values remain explicit.
"""

from pathlib import Path
from tempfile import TemporaryDirectory
import hashlib
import json

import geopandas as gpd
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "data/processed/planning_conflict_intervention"
REPORT = ROOT / "outputs/reports/planning_conflict_intervention"

GRID = BASE / "common_grid/18b_common_grid_integration_100m.gpkg"
QUEUES = BASE / "intervention_prioritization/18g_separate_planning_review_queues_100m.csv"
ASSIGN = BASE / "gn_hotspots/18i_01_grid_gn_assignment_100m.csv"
GN_GIS = BASE / "gn_hotspots/18i_01_gn_area_weighted_summary.gpkg"
HOT_GRID = BASE / "gn_hotspots/18i_02_spatial_conflict_hotspot_grid_membership_100m.csv"
HOT_GN = BASE / "gn_hotspots/18i_02_gn_hotspot_summary.csv"
HOT_GIS = BASE / "gn_hotspots/18i_02_spatial_conflict_hotspots.gpkg"
STABILITY = BASE / "intervention_prioritization/18h_screening_hotspot_grid_stability.csv"

QA_B = REPORT / "18b_common_grid_integration_qa.json"
QA_G = REPORT / "18g_separate_planning_review_queues_qa.json"
QA_I1 = REPORT / "18i_01_gn_area_weighted_summary_qa.json"
QA_I2 = REPORT / "18i_02_spatial_conflict_hotspots_qa.json"
QA_H = REPORT / "18h_planning_review_and_hotspot_sensitivity_qa.json"

OUT = BASE / "final_gis/18j_planning_conflict_final.gpkg"
OUT_QA = REPORT / "18j_planning_conflict_final_gis_qa.json"

SOURCE = [
    GRID, QUEUES, ASSIGN, GN_GIS, HOT_GRID,
    HOT_GN, HOT_GIS, STABILITY,
    QA_B, QA_G, QA_I1, QA_I2, QA_H,
]

EXPECTED = {
    "b": "PASS_COMMON_GRID_FINAL_EXPERT_AHP_INTEGRATION",
    "g": "PASS_SEPARATE_REVIEW_QUEUE_RANKINGS_SENSITIVITY_PENDING",
    "i1": "PASS_GN_AREA_WEIGHTED_INTEGRATION_WITH_COVERAGE_AUDIT",
    "i2": "PASS_EXPLORATORY_DCI_CP_SPATIAL_HOTSPOTS_WITH_GN_GAPS",
    "h": "PASS_REVIEW_AND_SPATIAL_HOTSPOT_SENSITIVITY_WITH_BASELINE_REPRODUCTION",
}

N = 9240
N_GN = 57
N_P = 4620
N_CP = 4609

AREA_KM2 = 88.82921581108494


# ============================================================
# 1. HELPERS
# ============================================================

def ensure(ok, label):

    if not bool(ok):
        raise ValueError(f"[FAIL] {label}")

    print(f"[PASS] {label}")


def sha(path):

    h = hashlib.sha256()

    with path.open("rb") as f:
        for block in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(block)

    return h.hexdigest()


def rel(path):

    return path.relative_to(ROOT).as_posix()


def nums(df, key):

    ensure(
        key in df.columns,
        f"Field present: {key}",
    )

    return pd.to_numeric(
        df[key],
        errors="coerce",
    ).to_numpy(dtype=float)


def unique(df, key, n, label):

    ensure(
        key in df
        and len(df) == n
        and df[key].notna().all()
        and df[key].is_unique,
        f"{label}: {n:,} unique IDs",
    )


def align(df, ids):

    ensure(
        set(df.grid_id) == set(ids),
        "Source grid identities match",
    )

    return (
        df.set_index("grid_id")
        .loc[ids]
        .reset_index()
    )


def nullable_bool(df, key, domain):

    raw = (
        df[key]
        .astype("string")
        .str.strip()
        .str.lower()
    )

    ensure(
        raw.dropna().isin(
            ["true", "false", "1", "0"]
        ).all(),
        f"{key}: valid boolean values",
    )

    ensure(
        np.array_equal(
            raw.notna().to_numpy(),
            domain,
        ),
        f"{key}: missingness follows valid index domain",
    )

    result = pd.Series(
        pd.NA,
        index=df.index,
        dtype="boolean",
    )

    result.loc[
        raw.isin(["true", "1"])
    ] = True

    result.loc[
        raw.isin(["false", "0"])
    ] = False

    return result


# ============================================================
# 2. VERIFY RELEASES
# ============================================================

def load_qa():

    print("\n[1] VERIFY ALL COMPLETED SOURCE RELEASES")

    q = {
        "b": json.loads(
            QA_B.read_text(encoding="utf-8")
        ),
        "g": json.loads(
            QA_G.read_text(encoding="utf-8")
        ),
        "i1": json.loads(
            QA_I1.read_text(encoding="utf-8")
        ),
        "i2": json.loads(
            QA_I2.read_text(encoding="utf-8")
        ),
        "h": json.loads(
            QA_H.read_text(encoding="utf-8")
        ),
    }

    for key, expected in EXPECTED.items():

        ensure(
            q[key].get("status") == expected,
            f"Stage {key} QA status verified",
        )

    hash_checks = [
        (q["b"]["outputs"]["gpkg_sha256"], GRID),
        (q["g"]["output"]["sha256"], QUEUES),
        (q["i1"]["outputs"][rel(ASSIGN)], ASSIGN),
        (q["i1"]["outputs"][rel(GN_GIS)], GN_GIS),
        (q["i2"]["outputs"][rel(HOT_GRID)], HOT_GRID),
        (q["i2"]["outputs"][rel(HOT_GN)], HOT_GN),
        (q["i2"]["outputs"][rel(HOT_GIS)], HOT_GIS),
        (q["h"]["outputs_sha256"][rel(STABILITY)], STABILITY),
    ]

    for expected, path in hash_checks:

        ensure(
            expected == sha(path),
            f"Source output hash verified: {path.name}",
        )

    ensure(
        q["i2"]["input_sha256"][rel(GRID)]
        == sha(GRID),
        "Hotspots trace to exact final common grid",
    )

    ensure(
        q["i2"]["input_sha256"][rel(QUEUES)]
        == sha(QUEUES),
        "Hotspots trace to exact separate review queues",
    )

    ensure(
        q["h"]["inputs_sha256"][rel(HOT_GRID)]
        == sha(HOT_GRID),
        "Sensitivity traces to exact baseline hotspot membership",
    )

    ensure(
        q["h"]["inputs_sha256"][rel(QUEUES)]
        == sha(QUEUES),
        "Sensitivity traces to exact review rankings",
    )

    ensure(
        q["h"]["review_rule_scenarios"] == 9
        and q["h"]["hotspot_scenarios_per_index"] == 9,
        "Nine review and nine hotspot scenarios per indicator",
    )

    return q


# ============================================================
# 3. MAIN WORKFLOW
# ============================================================

def main():

    print("=" * 76)
    print(
        "SECTION 05 | STEP 5.10 - "
        "FINAL QA AND GIS PACKAGING"
    )
    print("=" * 76)

    for p in SOURCE:

        if not p.is_file():
            raise FileNotFoundError(p)

    for p in [OUT, OUT_QA]:

        if p.exists():
            raise FileExistsError(
                f"Will not overwrite: {p}"
            )

    baseline_hashes = {
        rel(p): sha(p)
        for p in SOURCE
    }

    qa = load_qa()

    # ========================================================
    # 4. LOAD VERIFIED RESEARCH RESULTS
    # ========================================================

    print("\n[2] READ AND JOIN VERIFIED RESEARCH OUTPUTS")

    grid_src = gpd.read_file(
        GRID,
        layer="planning_conflict_common_grid_100m",
    )

    queue = pd.read_csv(QUEUES)

    assignments = pd.read_csv(
        ASSIGN,
        dtype={"gn_id": "string"},
    )

    hotspot_grid = pd.read_csv(HOT_GRID)

    stability = pd.read_csv(STABILITY)

    gn_src = gpd.read_file(
        GN_GIS,
        layer="gn_planning_evidence_summary",
    )

    gn_hotspot = pd.read_csv(
        HOT_GN,
        dtype={"gn_id": "string"},
    )

    clusters = gpd.read_file(
        HOT_GIS,
        layer="screened_spatial_groups",
    )

    for df, key, count, name in [
        (grid_src, "grid_id", N, "Original planning geometry"),
        (queue, "grid_id", N, "Review queues"),
        (assignments, "grid_id", N, "GN assignments"),
        (hotspot_grid, "grid_id", N, "Baseline hotspot membership"),
        (stability, "grid_id", N, "Grid sensitivity"),
        (gn_src, "gn_id", N_GN, "GN summary geometry"),
        (gn_hotspot, "gn_id", N_GN, "GN hotspot area summary"),
        (clusters, "cluster_id", 673, "Exploratory spatial groups"),
    ]:

        unique(
            df,
            key,
            count,
            name,
        )

    for df, name in [
        (grid_src, "Grid"),
        (gn_src, "GN"),
        (clusters, "Hotspot"),
    ]:

        ensure(
            df.crs is not None
            and df.crs.to_epsg() == 32644,
            f"{name}: EPSG:32644",
        )

        ensure(
            df.geometry.notna().all()
            and df.geometry.is_valid.all()
            and not df.geometry.is_empty.any(),
            f"{name}: valid geometries",
        )

    ids = grid_src.grid_id

    queue = align(queue, ids)
    assignments = align(assignments, ids)
    hotspot_grid = align(hotspot_grid, ids)
    stability = align(stability, ids)

    original_area = (
        grid_src.geometry.area.to_numpy(
            dtype=float
        )
    )

    ensure(
        np.all(original_area > 0)
        and np.isclose(
            original_area.sum() / 1e6,
            AREA_KM2,
            atol=0.001,
        ),
        "Grid covers documented KMC area",
    )

    # ========================================================
    # 5. SELECT FINAL GIS FIELDS
    # ========================================================

    queue_fields = [
        "grid_id",
        "pressure_domain_status",
        "review_rule_id",
        "review_stream",
        "review_action",
        "explanation_of_rule_trigger",
        "urbanization_pressure_score",
        "pressure_threshold_flag_frozen",
        "development_suitability_score",
        "development_conflict_index_0_1",
        "frozen_conservation_priority_0_1",
        "historical_context_type",
        "historical_robust_exposure_share_500m",
        "dci_rank_in_pressure_domain",
        "cp_rank_in_conservation_domain",
        "flagged_development_review_rank_by_dci",
        "upper_cp_conservation_review_rank_by_cp",
        "joint_review_rank_by_dci",
        "joint_review_rank_by_cp",
        "missing_conservation_evidence_rank_by_dci",
        "managed_growth_screen_rank_by_suitability",
    ]

    assignment_fields = [
        "grid_id",
        "gn_id",
        "gn_name",
        "gn_assignment_status",
        "gn_intersection_count",
        "gn_covered_fraction",
        "gn_uncovered_area_m2",
        "dominant_gn_fraction_of_grid",
    ]

    hotspot_fields = [
        "grid_id",
        "dci_screen_status",
        "dci_cluster_id",
        "dci_cluster_size_cells",
        "dci_candidate_hotspot",
        "cp_screen_status",
        "cp_cluster_id",
        "cp_cluster_size_cells",
        "cp_candidate_hotspot",
    ]

    stability_fields = [
        "grid_id",
        "review_rule_changes_in_8_stress_scenarios",
        "joint_concern_presence_in_9_rule_scenarios",
        "feasibility_presence_in_9_rule_scenarios",
        "dci_candidate_hotspot_presence_in_9_scenarios",
        "cp_candidate_hotspot_presence_in_9_scenarios",
        "dci_top_screen_presence_in_9_scenarios",
        "cp_top_screen_presence_in_9_scenarios",
    ]

    for df, fields, label in [
        (queue, queue_fields, "18G"),
        (assignments, assignment_fields, "GN assignment"),
        (hotspot_grid, hotspot_fields, "18I-02"),
        (stability, stability_fields, "18H"),
    ]:

        ensure(
            set(fields).issubset(df.columns),
            f"{label}: required fields available",
        )

    grid = grid_src[
        [
            "grid_id",
            "kmc_overlap_area_m2",
            "geometry",
        ]
    ].copy()

    for df, fields in [
        (queue, queue_fields),
        (assignments, assignment_fields),
        (hotspot_grid, hotspot_fields),
        (stability, stability_fields),
    ]:

        grid = grid.merge(
            df[fields],
            on="grid_id",
            how="left",
            validate="one_to_one",
        )

    grid = gpd.GeoDataFrame(
        grid,
        geometry="geometry",
        crs=grid_src.crs,
    )

    unique(
        grid,
        "grid_id",
        N,
        "Final GIS planning grids",
    )

    # ========================================================
    # 6. VERIFY MODEL DOMAINS AND SCORES
    # ========================================================

    print("\n[3] VALIDATE MODEL DOMAINS AND FINAL INDEX VALUES")

    p = nums(
        grid,
        "urbanization_pressure_score",
    )

    s = nums(
        grid,
        "development_suitability_score",
    )

    dci = nums(
        grid,
        "development_conflict_index_0_1",
    )

    cp = nums(
        grid,
        "frozen_conservation_priority_0_1",
    )

    p_ok = np.isfinite(p)
    cp_ok = np.isfinite(cp)

    ensure(
        p_ok.sum() == N_P
        and cp_ok.sum() == N_CP,
        "4,620 pressure cells; 4,609 CP cells",
    )

    ensure(
        np.array_equal(
            np.isfinite(dci),
            p_ok,
        )
        and np.all(
            ~cp_ok | p_ok
        ),
        "DCI and CP domains preserved without imputation",
    )

    ensure(
        np.isfinite(s).all()
        and np.all(
            (s >= 0) & (s <= 1)
        )
        and np.allclose(
            dci[p_ok],
            p[p_ok] * (1 - s[p_ok]),
            atol=1e-12,
            rtol=0,
        ),
        "Final expert-AHP suitability and DCI preserved",
    )

    ensure(
        np.allclose(
            s,
            nums(
                grid_src,
                "development_suitability_score",
            ),
            atol=1e-12,
            rtol=0,
        ),
        "Frozen suitability matches GIS",
    )

    ensure(
        np.allclose(
            cp,
            nums(
                grid_src,
                "primary_conservation_priority_0_1",
            ),
            equal_nan=True,
            atol=1e-12,
            rtol=0,
        ),
        "Conservation Priority preserved unchanged",
    )

    ensure(
        np.array_equal(
            grid.review_rule_id.eq(
                "R00_OUTSIDE_PRESSURE_DOMAIN"
            ).to_numpy(),
            ~p_ok,
        ),
        "Out-of-domain review labels remain explicit",
    )

    ensure(
        np.array_equal(
            grid.review_rule_id.eq(
                "R01_CONSERVATION_EVIDENCE_GAP"
            ).to_numpy(),
            p_ok & ~cp_ok,
        ),
        "Eleven conservation evidence gaps preserved",
    )

    ensure(
        int(
            grid.review_rule_id.eq(
                "R02_JOINT_CONCERN"
            ).sum()
        ) == 7
        and int(
            grid.review_rule_id.eq(
                "R05_MANAGED_GROWTH_FEASIBILITY"
            ).sum()
        ) == 1,
        "Seven joint concerns and one feasibility screen preserved",
    )

    # ========================================================
    # 7. VERIFY SPATIAL HOTSPOT MEMBERSHIP
    # ========================================================

    dci_hot = nullable_bool(
        grid,
        "dci_candidate_hotspot",
        p_ok,
    )

    cp_hot = nullable_bool(
        grid,
        "cp_candidate_hotspot",
        cp_ok,
    )

    grid["dci_candidate_hotspot"] = dci_hot
    grid["cp_candidate_hotspot"] = cp_hot

    ensure(
        int(dci_hot.fillna(False).sum()) == 135
        and int(cp_hot.fillna(False).sum()) == 81,
        "Baseline DCI 135 and CP 81 hotspot cells preserved",
    )

    ensure(
        int(
            grid.dci_screen_status.eq(
                "CANDIDATE_HOTSPOT"
            ).sum()
        ) == 135
        and int(
            grid.cp_screen_status.eq(
                "CANDIDATE_HOTSPOT"
            ).sum()
        ) == 81,
        "Cluster statuses match hotspot memberships",
    )

    ensure(
        np.array_equal(
            grid["dci_cluster_id"].notna().to_numpy(),
            (
                grid.dci_screen_status.ne(
                    "NOT_IN_TOP_DECILE"
                ).to_numpy()
                & p_ok
            ),
        ),
        "DCI cluster IDs exactly match screened cells",
    )

    ensure(
        np.array_equal(
            grid["cp_cluster_id"].notna().to_numpy(),
            (
                grid.cp_screen_status.ne(
                    "NOT_IN_TOP_DECILE"
                ).to_numpy()
                & cp_ok
            ),
        ),
        "CP cluster IDs exactly match screened cells",
    )

    # ========================================================
    # 8. PRESERVE GN GAPS AND SENSITIVITY
    # ========================================================

    print("\n[4] CHECK GN BOUNDARY GAP AND STABILITY LIMITATIONS")

    gn_gap = float(
        qa["i1"]["grid_area_uncovered_by_gn_km2"]
    )

    ensure(
        np.isclose(
            gn_gap,
            2.133336,
            atol=0.000001,
            rtol=0,
        ),
        "Known GN boundary gap retained",
    )

    ensure(
        int(
            grid.gn_assignment_status.eq(
                "GN_BOUNDARY_GAP_UNASSIGNED"
            ).sum()
        ) == 206,
        "206 grids without GN assignment remain explicit",
    )

    ensure(
        grid.loc[
            grid.gn_assignment_status.eq(
                "GN_BOUNDARY_GAP_UNASSIGNED"
            ),
            "gn_name",
        ].isna().all(),
        "Unassigned GN names remain missing",
    )

    ensure(
        np.isclose(
            nums(
                grid,
                "gn_uncovered_area_m2",
            ).sum() / 1e6,
            gn_gap,
            atol=0.000001,
            rtol=0,
        ),
        "Unattributed GN area balances original GN audit",
    )

    changed = nums(
        grid,
        "review_rule_changes_in_8_stress_scenarios",
    )

    dh_stability = nums(
        grid,
        "dci_candidate_hotspot_presence_in_9_scenarios",
    )

    ch_stability = nums(
        grid,
        "cp_candidate_hotspot_presence_in_9_scenarios",
    )

    ensure(
        np.array_equal(
            np.isfinite(changed),
            cp_ok,
        )
        and np.array_equal(
            np.isfinite(dh_stability),
            p_ok,
        )
        and np.array_equal(
            np.isfinite(ch_stability),
            cp_ok,
        ),
        "Sensitivity missingness follows supported score domains",
    )

    ensure(
        np.all(
            (changed[cp_ok] >= 0)
            & (changed[cp_ok] <= 8)
        )
        and np.all(
            (dh_stability[p_ok] >= 0)
            & (dh_stability[p_ok] <= 9)
        )
        and np.all(
            (ch_stability[cp_ok] >= 0)
            & (ch_stability[cp_ok] <= 9)
        ),
        "Sensitivity scenario-count ranges preserved",
    )

    hp_ranges = qa["h"][
        "hotspot_candidate_count_range_by_indicator"
    ]

    ensure(
        hp_ranges["DCI"][0] <= 36 <= hp_ranges["DCI"][1]
        and hp_ranges["CP"][0] <= 22 <= hp_ranges["CP"][1],
        "Baseline hotspots lie inside sensitivity ranges",
    )

    # ========================================================
    # 9. GN AND CLUSTER GIS
    # ========================================================

    print("\n[5] BUILD GN AND HOTSPOT GIS LAYERS")

    gn_src["gn_id"] = (
        gn_src["gn_id"]
        .astype("string")
        .str.strip()
    )

    gn_hotspot["gn_id"] = (
        gn_hotspot["gn_id"]
        .astype("string")
        .str.strip()
    )

    gn_fields = [
        "gn_id",
        "dci_hotspot_gn_area_ha",
        "dci_hotspot_share_of_gn_pct",
        "dci_hotspot_cluster_count",
        "cp_hotspot_gn_area_ha",
        "cp_hotspot_share_of_gn_pct",
        "cp_hotspot_cluster_count",
    ]

    ensure(
        set(gn_fields).issubset(
            gn_hotspot.columns
        ),
        "GN hotspot fields present",
    )

    ensure(
        set(gn_src.gn_id)
        == set(gn_hotspot.gn_id),
        "GN IDs match hotspot summary",
    )

    gn = gn_src.merge(
        gn_hotspot[gn_fields],
        on="gn_id",
        how="left",
        validate="one_to_one",
    )

    unique(
        gn,
        "gn_id",
        N_GN,
        "Final GN summary GIS",
    )

    expected_clusters = {
        "DEVELOPMENT_CONFLICT": 36,
        "CONSERVATION_PRIORITY": 22,
    }

    for indicator, total in expected_clusters.items():

        chosen = clusters.loc[
            clusters.index_type.eq(indicator)
        ]

        ensure(
            int(
                chosen.is_candidate_hotspot.sum()
            ) == total,
            f"{indicator}: baseline cluster count preserved",
        )

    ensure(
        int(
            clusters.is_candidate_hotspot.sum()
        ) == 58,
        "58 candidate hotspot clusters; small groups retained",
    )

    ensure(
        int(
            clusters.grid_count.sum()
        ) == 462 + 461,
        "All 923 separately screened grid memberships accounted for",
    )

    grid["is_official_planning_classification"] = False
    gn["is_official_planning_classification"] = False
    clusters["is_statistically_significant_hotspot"] = False

    # ========================================================
    # 10. EXPORT THREE GIS LAYERS
    # ========================================================

    print("\n[6] WRITE CONSOLIDATED GPKG AND VALIDATE")

    OUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUT_QA.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with TemporaryDirectory(
        prefix="18j_build_",
        dir=OUT.parent,
    ) as td:

        temp = Path(td) / OUT.name

        layers = [
            (
                "planning_grid_100m",
                grid,
                N,
                "grid_id",
            ),
            (
                "gn_planning_summary",
                gn,
                N_GN,
                "gn_id",
            ),
            (
                "exploratory_spatial_groups",
                clusters,
                673,
                "cluster_id",
            ),
        ]

        for layer_name, frame, _, _ in layers:

            frame.to_file(
                temp,
                layer=layer_name,
                driver="GPKG",
                engine="pyogrio",
            )

            print(
                f"[PASS] Wrote layer: {layer_name}"
            )

        for layer_name, frame, count, key in layers:

            readback = gpd.read_file(
                temp,
                layer=layer_name,
            )

            unique(
                readback,
                key,
                count,
                f"Roundtrip layer {layer_name}",
            )

            ensure(
                readback.crs is not None
                and readback.crs.to_epsg() == 32644
                and readback.geometry.is_valid.all(),
                f"{layer_name}: valid roundtrip GIS",
            )

            ensure(
                set(readback[key].astype(str))
                == set(frame[key].astype(str)),
                f"{layer_name}: all identifiers preserved",
            )

            if layer_name == "planning_grid_100m":

                for col in [
                    "development_conflict_index_0_1",
                    "frozen_conservation_priority_0_1",
                    "urbanization_pressure_score",
                ]:

                    ensure(
                        np.allclose(
                            nums(readback, col),
                            nums(frame, col),
                            equal_nan=True,
                            atol=1e-12,
                            rtol=0,
                        ),
                        f"GPKG preserves {col} and missingness",
                    )

                ensure(
                    int(
                        readback.review_rule_id.eq(
                            "R02_JOINT_CONCERN"
                        ).sum()
                    ) == 7,
                    "GPKG retains seven joint-concern cells",
                )

            if layer_name == "exploratory_spatial_groups":

                ensure(
                    int(
                        readback.is_candidate_hotspot.sum()
                    ) == 58,
                    "GPKG retains all 58 candidate clusters",
                )

        ensure(
            not OUT.exists()
            and not OUT_QA.exists(),
            "No existing final output overwritten",
        )

        temp.replace(OUT)

    # ========================================================
    # 11. VERIFY SOURCES
    # ========================================================

    print("\n[7] VERIFY INPUTS AND GENERATE FINAL PACKAGE QA")

    for path in SOURCE:

        ensure(
            sha(path) == baseline_hashes[rel(path)],
            f"Unmodified source: {path.name}",
        )

    # ========================================================
    # 12. FINAL GIS PACKAGE QA REPORT
    # ========================================================

    report = {

        "section":
            "05_planning_conflict_intervention",

        "stage":
            "18j_final_qa_and_gis_packaging",

        "status": (
            "PASS_SECTION05_GIS_PACKAGE_WITH_"
            "DOCUMENTED_LIMITATIONS_DASHBOARD_PENDING"
        ),

        "grid_cells": N,

        "gn_divisions": N_GN,

        "pressure_domain_cells": N_P,

        "conservation_common_domain_cells": N_CP,

        "gn_boundary_uncovered_km2": gn_gap,

        "gn_unassigned_grid_count": 206,

        "baseline_dci_candidate_hotspots": 36,

        "baseline_cp_candidate_hotspots": 22,

        "baseline_dci_hotspot_cells": 135,

        "baseline_cp_hotspot_cells": 81,

        "sensitivity_dci_hotspot_count_range":
            hp_ranges["DCI"],

        "sensitivity_cp_hotspot_count_range":
            hp_ranges["CP"],

        "planning_rule_sensitivity_joint_concern_range":
            qa["h"]["stress_review_joint_concern_count_range"],

        "planning_rule_sensitivity_feasibility_range":
            qa["h"]["stress_review_managed_growth_count_range"],

        "hotspot_result_interpretation": (
            "PARAMETER_SENSITIVE_EXPLORATORY_GROUPS_"
            "NOT_STATISTICAL_SIGNIFICANCE"
        ),

        "review_route_interpretation":
            "EXPLORATORY_SCREENING_NOT_OFFICIAL_POLICY",

        "pressure_is_calibrated_probability": False,

        "dci_and_cp_combined": False,

        "expert_ahp_or_frozen_model_modified": False,

        "input_files_modified": False,

        "dashboard_exports_pending": True,

        "limitations": [
            (
                "DCI is defined only for 4,620 "
                "pressure-domain grids."
            ),
            (
                "Conservation Priority is available for "
                "4,609 grids and already contains pressure."
            ),
            (
                "The original screening gives 36 DCI and "
                "22 conservation hotspots, but their "
                "counts are highly parameter-sensitive."
            ),
            (
                "Hotspots are exploratory groups, not "
                "statistically significant hotspots."
            ),
            (
                "Screening thresholds are not official "
                "policy or legal classifications."
            ),
            (
                "GN boundaries do not cover 2.133336 "
                "square kilometres of the study area."
            ),
            (
                "GN statistics use fractional area "
                "intersections, while popup GN names "
                "use dominant intersection."
            ),
            (
                "Development Suitability uses expert-AHP "
                "road proximity, road density and slope; "
                "it is not comprehensive legal, flood "
                "or drainage suitability."
            ),
            (
                "No development approval or official "
                "zoning decision is inferred."
            ),
        ],

        "inputs_sha256":
            baseline_hashes,

        "output": {
            "gpkg": rel(OUT),
            "sha256": sha(OUT),
            "layers": [
                "planning_grid_100m",
                "gn_planning_summary",
                "exploratory_spatial_groups",
            ],
        },
    }

    OUT_QA.write_text(
        json.dumps(
            report,
            indent=2,
        ),
        encoding="utf-8",
    )

    # ========================================================
    # 13. FINAL TERMINAL SUMMARY
    # ========================================================

    print("\n" + "=" * 76)

    print(report["status"])

    print("Final grid cells:", N)
    print("Final GN divisions:", N_GN)
    print("Exploratory spatial groups:", len(clusters))
    print("DCI / CP baseline hotspots: 36 / 22")

    print(
        "GN boundary gap:",
        f"{gn_gap:.6f} km2",
    )

    print("Hotspot sensitivity remains material: YES")
    print("Model sources modified: NO")
    print("Backend-ready exports complete: NO - NEXT STEP 5.11")

    print("\nGeoPackage:", OUT)
    print("QA report:", OUT_QA)

    print("=" * 76)


if __name__ == "__main__":
    main()
