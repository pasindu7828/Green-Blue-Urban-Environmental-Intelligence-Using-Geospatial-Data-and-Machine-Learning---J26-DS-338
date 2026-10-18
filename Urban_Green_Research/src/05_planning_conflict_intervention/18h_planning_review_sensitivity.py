
"""
J26-DS-338 | Section 05 | Step 5.9 (18H)
COMBINED SCREENING AND SPATIAL-HOTSPOT SENSITIVITY

Review rules: suitability low/high and frozen CP top ranks at 20/25/30%.
Hotspots: top 5/10/15% of DCI or frozen CP; min edge-connected
component size 2/3/5. Baseline is 10% and minimum 3 cells.

Descriptive research sensitivity, NOT official classifications or
statistical hotspot significance. Frozen model outputs are not changed.
"""

from collections import defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory
import hashlib
import json
import math

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely import force_2d


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "data/processed/planning_conflict_intervention"
REPORT = ROOT / "outputs/reports/planning_conflict_intervention"

D = BASE / "conflict_typology/18d_planning_conflict_typology_100m.csv"
F = BASE / "planning_intervention_rules/18f_explainable_planning_interventions_100m.csv"
G = BASE / "intervention_prioritization/18g_separate_planning_review_queues_100m.csv"
H = BASE / "gn_hotspots/18i_02_spatial_conflict_hotspot_grid_membership_100m.csv"
GIS = BASE / "common_grid/18b_common_grid_integration_100m.gpkg"

Q_D = REPORT / "18d_planning_conflict_typology_qa.json"
Q_F = REPORT / "18f_explainable_planning_interventions_qa.json"
Q_G = REPORT / "18g_separate_planning_review_queues_qa.json"
Q_H = REPORT / "18i_02_spatial_conflict_hotspots_qa.json"
Q_GIS = REPORT / "18b_common_grid_integration_qa.json"

OUT_REVIEW = BASE / "intervention_prioritization/18h_screening_sensitivity_scenarios.csv"
OUT_HOT = BASE / "intervention_prioritization/18h_hotspot_sensitivity_scenarios.csv"
OUT_GRID = BASE / "intervention_prioritization/18h_screening_hotspot_grid_stability.csv"
OUT_QA = REPORT / "18h_planning_review_and_hotspot_sensitivity_qa.json"

N, NP, NCP = 9240, 4620, 4609

RULE_PCTS = (20, 25, 30)
HOT_PCTS = (5, 10, 15)
MIN_SIZES = (2, 3, 5)

EDGE_M = 0.10
RULES = [f"R{i:02d}" for i in range(10)]


def must(condition, label):

    ok = bool(condition)

    print(f"[{'PASS' if ok else 'FAIL'}] {label}")

    if not ok:
        raise ValueError(f"18H validation failed: {label}")


def hash_file(path):

    h = hashlib.sha256()

    with path.open("rb") as fp:
        for block in iter(lambda: fp.read(1024 * 1024), b""):
            h.update(block)

    return h.hexdigest()


def rel(path):
    return path.relative_to(ROOT).as_posix()


def vec(df, key):

    if key not in df:
        raise KeyError(key)

    return pd.to_numeric(
        df[key], errors="coerce"
    ).to_numpy(dtype=float)


def unique(df, label):

    must(
        len(df) == N
        and df.grid_id.notna().all()
        and df.grid_id.is_unique,
        f"{label}: 9,240 unique IDs",
    )


def top_mask(values, ids, domain, count, highest=True):

    idx = np.flatnonzero(domain)

    must(
        0 < count <= len(idx)
        and np.isfinite(values[idx]).all(),
        f"Screen has {count} valid candidates",
    )

    keys = -values[idx] if highest else values[idx]

    selected = idx[
        np.lexsort((ids[idx], keys))[:count]
    ]

    result = np.zeros(N, dtype=bool)
    result[selected] = True

    return result


def routes_for(domain, cp_domain, flagged, low, high, cp_upper):

    common = domain & cp_domain

    joint = flagged & low & cp_upper

    managed = (
        flagged & high & common & ~cp_upper
    )

    masks = [
        ("R00", ~domain),
        ("R01", domain & ~cp_domain),
        ("R02", common & joint),
        (
            "R03",
            common & flagged & cp_upper & ~joint,
        ),
        (
            "R04",
            common & flagged & low & ~cp_upper,
        ),
        ("R05", common & managed),
        (
            "R06",
            common & flagged
            & ~cp_upper & ~low & ~managed,
        ),
        (
            "R07",
            common & ~flagged & cp_upper,
        ),
        (
            "R08",
            common & ~flagged & low & ~cp_upper,
        ),
        (
            "R09",
            common & ~flagged & ~cp_upper & ~low,
        ),
    ]

    codes = np.full(
        N, "UNASSIGNED", dtype=object
    )

    for code, mask in masks:

        must(
            not np.any(
                mask & (codes != "UNASSIGNED")
            ),
            f"No conflicting rule assignment: {code}",
        )

        codes[mask] = code

    must(
        np.all(codes != "UNASSIGNED"),
        "All rule cases covered",
    )

    return codes, joint, managed


def spatial_edges(geometry, rank, cap):

    """
    Build edge-sharing graph up to
    the largest hotspot screening extent.
    """

    global_idx = np.flatnonzero(
        np.isfinite(rank) & (rank <= cap)
    )

    subset = (
        geometry.iloc[global_idx]
        .reset_index(drop=True)
    )

    index = subset.sindex
    geoms = subset.geometry.to_list()

    edges = []

    for i, geom in enumerate(geoms):

        for j0 in index.query(
            geom, predicate="intersects"
        ):

            j = int(j0)

            if j <= i:
                continue

            shared_length = (
                geom.boundary.intersection(
                    geoms[j].boundary
                ).length
            )

            if shared_length > EDGE_M:

                edges.append(
                    (
                        int(global_idx[i]),
                        int(global_idx[j]),
                    )
                )

    return edges


def components_for(screened, edges, area, minimum):

    idx = np.flatnonzero(screened)

    parent = {
        int(i): int(i) for i in idx
    }

    sizes = {
        int(i): 1 for i in idx
    }

    def find(i):

        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]

        return i

    for a, b in edges:

        if not (
            screened[a] and screened[b]
        ):
            continue

        ra, rb = find(a), find(b)

        if ra == rb:
            continue

        if sizes[ra] < sizes[rb]:
            ra, rb = rb, ra

        parent[rb] = ra
        sizes[ra] += sizes[rb]

    groups = defaultdict(list)

    for i in idx:
        groups[find(int(i))].append(int(i))

    candidate = np.zeros(
        N, dtype=bool
    )

    hot_count = 0
    small_count = 0

    for members in groups.values():

        if len(members) >= minimum:
            candidate[members] = True
            hot_count += 1
        else:
            small_count += 1

    return {
        "candidate": candidate,
        "candidate_groups": hot_count,
        "small_groups": small_count,
        "candidate_cells": int(candidate.sum()),
        "candidate_area_km2": float(
            area[candidate].sum() / 1e6
        ),
        "all_groups": len(groups),
    }


def read_boolean_mask(df, name, expected_domain):

    s = (
        df[name]
        .astype("string")
        .str.strip()
        .str.lower()
    )

    must(
        s.dropna().isin(
            ["true", "false", "1", "0"]
        ).all(),
        f"{name}: boolean values valid",
    )

    must(
        np.array_equal(
            s.notna().to_numpy(),
            expected_domain,
        ),
        f"{name}: missingness matches domain",
    )

    return (
        s.isin(["true", "1"])
        .fillna(False)
        .to_numpy(dtype=bool)
    )


def main():

    print("=" * 78)
    print(
        "SECTION 05 | STEP 5.9 - "
        "REVIEW & HOTSPOT SENSITIVITY"
    )
    print("=" * 78)

    sources = (
        D, F, G, H, GIS,
        Q_D, Q_F, Q_G, Q_H, Q_GIS,
    )

    outputs = (
        OUT_REVIEW, OUT_HOT, OUT_GRID, OUT_QA,
    )

    for p in sources:
        if not p.is_file():
            raise FileNotFoundError(p)

    for p in outputs:
        if p.exists():
            raise FileExistsError(
                f"Will not overwrite previous result: {p}"
            )

    inputs = {
        rel(p): hash_file(p)
        for p in sources
    }

    # ========================================================
    # 1. VERIFY PREVIOUS RELEASES
    # ========================================================

    print("\n[1] VERIFY PREVIOUS RELEASES")

    qa = {
        "18d": json.loads(
            Q_D.read_text(encoding="utf-8")
        ),
        "18f": json.loads(
            Q_F.read_text(encoding="utf-8")
        ),
        "18g": json.loads(
            Q_G.read_text(encoding="utf-8")
        ),
        "18i02": json.loads(
            Q_H.read_text(encoding="utf-8")
        ),
        "18b": json.loads(
            Q_GIS.read_text(encoding="utf-8")
        ),
    }

    statuses = {
        "18d": (
            "PASS_EXPLORATORY_PLANNING_TYPOLOGY_"
            "WITH_DOMAIN_LIMITS"
        ),
        "18f": (
            "PASS_EXPLAINABLE_PLANNING_REVIEW_RULES_"
            "PRIORITIZATION_PENDING"
        ),
        "18g": (
            "PASS_SEPARATE_REVIEW_QUEUE_RANKINGS_"
            "SENSITIVITY_PENDING"
        ),
        "18i02": (
            "PASS_EXPLORATORY_DCI_CP_SPATIAL_"
            "HOTSPOTS_WITH_GN_GAPS"
        ),
        "18b": (
            "PASS_COMMON_GRID_FINAL_EXPERT_"
            "AHP_INTEGRATION"
        ),
    }

    for key, expected in statuses.items():

        must(
            qa[key].get("status") == expected,
            f"{key} QA status",
        )

    for key, csv in (
        ("18d", D),
        ("18f", F),
        ("18g", G),
    ):

        must(
            qa[key]["output"]["sha256"]
            == hash_file(csv),
            f"{key} source hash verified",
        )

    must(
        qa["18b"]["outputs"]["gpkg_sha256"]
        == hash_file(GIS),
        "18B original grid GIS hash verified",
    )

    must(
        qa["18i02"]["outputs"][rel(H)]
        == hash_file(H),
        "18I-02 grid membership hash verified",
    )

    for source in (G, GIS):

        must(
            qa["18i02"]["input_sha256"].get(
                rel(source)
            ) == hash_file(source),
            (
                "18I-02 derives from exact "
                f"{source.name}"
            ),
        )

    # ========================================================
    # 2. LOAD AND ALIGN RESEARCH DATA
    # ========================================================

    print(
        "\n[2] ALIGN FROZEN GRID SCORES "
        "AND HOTSPOT MEMBERSHIP"
    )

    g = pd.read_csv(G)
    d = pd.read_csv(D)
    f = pd.read_csv(F)
    h = pd.read_csv(H)

    geom = gpd.read_file(
        GIS,
        layer="planning_conflict_common_grid_100m",
    )

    for name, df in (
        ("18G", g),
        ("18D", d),
        ("18F", f),
        ("18I-02", h),
        ("18B GIS", geom),
    ):

        unique(df, name)

        must(
            set(df.grid_id) == set(g.grid_id),
            f"{name}: matching grid IDs",
        )

    d = (
        d.set_index("grid_id")
        .loc[g.grid_id]
        .reset_index()
    )

    f = (
        f.set_index("grid_id")
        .loc[g.grid_id]
        .reset_index()
    )

    h = (
        h.set_index("grid_id")
        .loc[g.grid_id]
        .reset_index()
    )

    geom = (
        geom.set_index("grid_id")
        .loc[g.grid_id]
        .reset_index()
    )

    geom = gpd.GeoDataFrame(
        geom,
        geometry="geometry",
        crs="EPSG:32644",
    )

    must(
        geom.crs.to_epsg() == 32644,
        "GIS coordinates are EPSG:32644",
    )

    must(
        geom.geometry.notna().all()
        and geom.geometry.is_valid.all()
        and not geom.geometry.is_empty.any(),
        "Grid geometry valid",
    )

    geom["geometry"] = [
        force_2d(x)
        for x in geom.geometry
    ]

    area = geom.geometry.area.to_numpy(
        dtype=float
    )

    must(
        np.all(area > 0)
        and np.isclose(
            area.sum() / 1e6,
            88.82921581108494,
            atol=.001,
            rtol=0,
        ),
        "All grid areas verified",
    )

    ids = g.grid_id.astype(str).to_numpy()

    p = vec(
        g, "urbanization_pressure_score"
    )

    s = vec(
        g, "development_suitability_score"
    )

    cp = vec(
        g, "frozen_conservation_priority_0_1"
    )

    dci = vec(
        g, "development_conflict_index_0_1"
    )

    flags = vec(
        g, "pressure_threshold_flag_frozen"
    )

    p_valid = np.isfinite(p)
    cp_valid = np.isfinite(cp)

    flagged = p_valid & (flags == 1)

    must(
        p_valid.sum() == NP
        and cp_valid.sum() == NCP
        and flagged.sum() == 255,
        (
            "4,620 pressure, 4,609 conservation, "
            "255 pressure-flagged"
        ),
    )

    must(
        np.all(~cp_valid | p_valid)
        and np.array_equal(
            np.isfinite(dci), p_valid
        ),
        "Missingness domains preserved",
    )

    must(
        np.isfinite(s).all()
        and np.all(
            (s >= 0) & (s <= 1)
        )
        and np.allclose(
            dci[p_valid],
            p[p_valid] * (1 - s[p_valid]),
            atol=1e-12,
            rtol=0,
        ),
        "Expert-AHP DCI unchanged",
    )

    must(
        np.array_equal(
            f.review_rule_id.astype(str).to_numpy(),
            g.review_rule_id.astype(str).to_numpy(),
        ),
        "Frozen rule assignments match Step 18G",
    )

    recorded = (
        f.review_rule_id.astype(str)
        .str.extract(
            r"^(R\d\d)_",
            expand=False,
        )
    )

    must(
        recorded.notna().all()
        and recorded.isin(RULES).all(),
        "Review rules have recognized R00..R09 identifiers",
    )

    recorded = recorded.to_numpy(dtype=str)

    # ========================================================
    # 3. VERIFY ORIGINAL PLANNING RULES
    # ========================================================

    print(
        "\n[3] ORIGINAL RULE-SET REPRODUCTION"
    )

    low25 = top_mask(
        s, ids, p_valid,
        math.ceil(NP * .25),
        highest=False,
    )

    high25 = top_mask(
        s, ids, p_valid,
        math.ceil(NP * .25),
        highest=True,
    )

    cp25 = top_mask(
        cp, ids, cp_valid,
        math.ceil(NCP * .25),
        highest=True,
    )

    must(
        np.array_equal(
            low25,
            d.relative_suitability_screen.eq(
                "LOWEST_25_PERCENT_DOMAIN_RELATIVE"
            ).to_numpy(),
        )
        and np.array_equal(
            high25,
            d.relative_suitability_screen.eq(
                "HIGHEST_25_PERCENT_DOMAIN_RELATIVE"
            ).to_numpy(),
        )
        and np.array_equal(
            cp25,
            d.relative_frozen_cp_screen.eq(
                "HIGHEST_25_PERCENT_COMMON_DOMAIN"
            ).to_numpy(),
        ),
        "Original quartile memberships exactly reproduced",
    )

    (
        base_routes,
        base_joint,
        base_managed,
    ) = routes_for(
        p_valid,
        cp_valid,
        flagged,
        low25,
        high25,
        cp25,
    )

    must(
        np.array_equal(
            base_routes, recorded
        )
        and base_joint.sum() == 7
        and base_managed.sum() == 1,
        (
            "Original rules, 7 joint and "
            "1 feasibility reproduced"
        ),
    )

    # ========================================================
    # 4. REVIEW RULE SENSITIVITY
    # ========================================================

    print(
        "\n[4] REVIEW-RULE SENSITIVITY: "
        "NINE CUTOFF COMBINATIONS"
    )

    review_rows = []

    change_counter = np.zeros(
        N, dtype=int
    )

    joint_counter = np.zeros(
        N, dtype=int
    )

    managed_counter = np.zeros(
        N, dtype=int
    )

    for sp in RULE_PCTS:

        low = top_mask(
            s,
            ids,
            p_valid,
            math.ceil(NP * sp / 100),
            highest=False,
        )

        high = top_mask(
            s,
            ids,
            p_valid,
            math.ceil(NP * sp / 100),
            highest=True,
        )

        must(
            not (low & high).any(),
            f"Suitability {sp}% extremes disjoint",
        )

        for cp_pct in RULE_PCTS:

            cup = top_mask(
                cp,
                ids,
                cp_valid,
                math.ceil(NCP * cp_pct / 100),
                highest=True,
            )

            routes, joint, managed = routes_for(
                p_valid,
                cp_valid,
                flagged,
                low,
                high,
                cup,
            )

            baseline = (
                sp == 25
                and cp_pct == 25
            )

            changed = (
                routes != recorded
            )

            must(
                not changed[~cp_valid].any(),
                "Unsupported evidence routes unchanged",
            )

            if baseline:

                must(
                    not changed.any(),
                    "25/25 rules exactly identical to baseline",
                )

            else:

                change_counter += (
                    changed.astype(int)
                )

            joint_counter += joint.astype(int)
            managed_counter += managed.astype(int)

            review_rows.append({

                "scenario":
                    f"S{sp:02d}_CP{cp_pct:02d}",

                "baseline":
                    baseline,

                "suitability_extreme_pct":
                    sp,

                "cp_upper_pct":
                    cp_pct,

                "flagged_pressure_cells_frozen":
                    255,

                "joint_concern_cells":
                    int(joint.sum()),

                "managed_growth_screen_cells":
                    int(managed.sum()),

                "retained_baseline_joint_cells":
                    int(
                        (joint & base_joint).sum()
                    ),

                "changed_review_rule_cells":
                    int(changed.sum()),

                "changed_rule_pct_common_domain":
                    float(
                        100 * changed.sum() / NCP
                    ),

                **{
                    f"count_{r}":
                        int((routes == r).sum())
                    for r in RULES
                },
            })

    review = pd.DataFrame(
        review_rows
    )

    must(
        len(review) == 9
        and review.baseline.sum() == 1,
        "Nine rule scenarios including exact baseline",
    )

    # ========================================================
    # 5. SPATIAL HOTSPOT SENSITIVITY
    # ========================================================

    print(
        "\n[5] BUILD EDGE-ADJACENCY "
        "ON MAXIMUM 15% SCREEN"
    )

    hotspot_rows = []
    hotspot_memberships = {}

    for (
        name,
        metric,
        rankcol,
        status_prefix,
        previous_key,
    ) in (

        (
            "DCI",
            dci,
            "dci_rank_in_pressure_domain",
            "dci",
            "dci",
        ),

        (
            "CP",
            cp,
            "cp_rank_in_conservation_domain",
            "cp",
            "conservation_priority",
        ),

    ):

        ranks = vec(
            g, rankcol
        )

        domain = np.isfinite(metric)
        n_valid = int(domain.sum())

        must(
            np.array_equal(
                np.isfinite(ranks),
                domain,
            )
            and np.array_equal(
                np.sort(
                    ranks[domain].astype(int)
                ),
                np.arange(
                    1, n_valid + 1
                ),
            ),
            f"{name}: frozen full-domain ranks valid",
        )

        cap = math.ceil(
            n_valid * .15
        )

        edges = spatial_edges(
            geom,
            ranks,
            cap,
        )

        print(
            f"{name}: max-screen nodes={cap}, "
            f"shared-edge links={len(edges)}"
        )

        historical_hot = read_boolean_mask(
            h,
            f"{status_prefix}_candidate_hotspot",
            domain,
        )

        historical_screen = (
            h[
                f"{status_prefix}_screen_status"
            ]
            .isin([
                "CANDIDATE_HOTSPOT",
                "SMALL_SCREENED_GROUP",
            ])
            .to_numpy()
        )

        all_candidates = np.zeros(
            N, dtype=int
        )

        all_screened = np.zeros(
            N, dtype=int
        )

        for pct in HOT_PCTS:

            selected = (
                domain
                & (
                    ranks
                    <= math.ceil(
                        n_valid * pct / 100
                    )
                )
            )

            for minimum in MIN_SIZES:

                analysis = components_for(
                    selected,
                    edges,
                    area,
                    minimum,
                )

                hotmask = analysis["candidate"]

                baseline = (
                    pct == 10
                    and minimum == 3
                )

                if baseline:

                    must(
                        np.array_equal(
                            selected,
                            historical_screen,
                        ),
                        (
                            f"{name}: original top-10% "
                            "selection reproduced"
                        ),
                    )

                    must(
                        np.array_equal(
                            hotmask,
                            historical_hot,
                        ),
                        (
                            f"{name}: original candidate "
                            "hotspot cells reproduced"
                        ),
                    )

                    reference = (
                        qa["18i02"][previous_key]
                    )

                    must(
                        analysis["candidate_groups"]
                        == reference["candidate_hotspot_count"]
                        and analysis["candidate_cells"]
                        == reference["candidate_hotspot_cells"]
                        and abs(
                            analysis["candidate_area_km2"]
                            - reference["candidate_hotspot_area_km2"]
                        ) < 1e-7,
                        (
                            f"{name}: original hotspot "
                            "count/cells/area reproduced"
                        ),
                    )

                all_candidates += hotmask.astype(int)
                all_screened += selected.astype(int)

                hotspot_rows.append({

                    "indicator":
                        name,

                    "top_domain_pct":
                        pct,

                    "minimum_connected_cells":
                        minimum,

                    "baseline":
                        baseline,

                    "screened_cells":
                        int(selected.sum()),

                    "candidate_hotspots":
                        analysis["candidate_groups"],

                    "candidate_hotspot_cells":
                        analysis["candidate_cells"],

                    "candidate_hotspot_area_km2":
                        analysis["candidate_area_km2"],

                    "small_screened_groups":
                        analysis["small_groups"],

                    "all_screened_groups":
                        analysis["all_groups"],

                    "baseline_hotspot_cells_retained":
                        int(
                            (hotmask & historical_hot).sum()
                        ),

                    "baseline_hotspot_cells_lost":
                        int(
                            (~hotmask & historical_hot).sum()
                        ),

                    "new_hotspot_cells":
                        int(
                            (hotmask & ~historical_hot).sum()
                        ),
                })

            print(
                f"{name}, top {pct}%: completed "
                "minimum sizes 2/3/5"
            )

        hotspot_memberships[name] = (
            all_candidates,
            all_screened,
        )

    hotspots = pd.DataFrame(
        hotspot_rows
    )

    must(
        len(hotspots) == 18
        and hotspots.groupby(
            "indicator"
        ).baseline.sum().eq(1).all(),
        (
            "18 hotspot scenarios; "
            "one exact baseline for each index"
        ),
    )

    # ========================================================
    # 6. COMBINE GRID-LEVEL STABILITY
    # ========================================================

    print(
        "\n[6] BUILD GRID-LEVEL SCENARIO STABILITY"
    )

    dci_cand, dci_sel = (
        hotspot_memberships["DCI"]
    )

    cp_cand, cp_sel = (
        hotspot_memberships["CP"]
    )

    stable = pd.DataFrame({

        "grid_id":
            g.grid_id,

        "original_review_rule_id":
            g.review_rule_id,

        "review_rule_changes_in_8_stress_scenarios":
            pd.Series(
                change_counter
            ).where(cp_valid).astype("Int64"),

        "joint_concern_presence_in_9_rule_scenarios":
            pd.Series(
                joint_counter
            ).where(cp_valid).astype("Int64"),

        "feasibility_presence_in_9_rule_scenarios":
            pd.Series(
                managed_counter
            ).where(cp_valid).astype("Int64"),

        "dci_candidate_hotspot_presence_in_9_scenarios":
            pd.Series(
                dci_cand
            ).where(p_valid).astype("Int64"),

        "cp_candidate_hotspot_presence_in_9_scenarios":
            pd.Series(
                cp_cand
            ).where(cp_valid).astype("Int64"),

        "dci_top_screen_presence_in_9_scenarios":
            pd.Series(
                dci_sel
            ).where(p_valid).astype("Int64"),

        "cp_top_screen_presence_in_9_scenarios":
            pd.Series(
                cp_sel
            ).where(cp_valid).astype("Int64"),
    })

    unique(
        stable,
        "Combined stability",
    )

    must(
        not change_counter[~cp_valid].any()
        and np.all(
            dci_cand[~p_valid] == 0
        )
        and np.all(
            cp_cand[~cp_valid] == 0
        ),
        (
            "No inference outside supported "
            "pressure/conservation domains"
        ),
    )

    must(
        int(joint_counter.sum())
        == int(
            review.joint_concern_cells.sum()
        ),
        (
            "Review scenario joint counts reconcile "
            "to grid memberships"
        ),
    )

    must(
        int(dci_cand.sum())
        == int(
            hotspots.loc[
                hotspots.indicator.eq("DCI"),
                "candidate_hotspot_cells",
            ].sum()
        ),
        "DCI hotspot scenario counts reconcile",
    )

    must(
        int(cp_cand.sum())
        == int(
            hotspots.loc[
                hotspots.indicator.eq("CP"),
                "candidate_hotspot_cells",
            ].sum()
        ),
        "Conservation hotspot scenario counts reconcile",
    )

    # ========================================================
    # 7. EXPORT
    # ========================================================

    print(
        "\n[7] EXPORT THREE TABLES AND VALIDATE"
    )

    OUT_REVIEW.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUT_QA.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with TemporaryDirectory(
        prefix="18h_build_",
        dir=OUT_REVIEW.parent,
    ) as tmp:

        temp = Path(tmp)

        paths = [
            (
                review,
                temp / OUT_REVIEW.name,
                OUT_REVIEW,
            ),
            (
                hotspots,
                temp / OUT_HOT.name,
                OUT_HOT,
            ),
            (
                stable,
                temp / OUT_GRID.name,
                OUT_GRID,
            ),
        ]

        for df, src, dest in paths:

            df.to_csv(
                src,
                index=False,
            )

            must(
                len(pd.read_csv(src)) == len(df),
                f"Roundtrip count: {dest.name}",
            )

        check = pd.read_csv(
            temp / OUT_GRID.name
        )

        must(
            check.grid_id.equals(
                stable.grid_id
            ),
            "Grid-level stability IDs preserved",
        )

        must(
            pd.read_csv(
                temp / OUT_HOT.name
            ).baseline.sum() == 2,
            (
                "Export preserves two spatial "
                "baseline scenarios"
            ),
        )

        must(
            pd.read_csv(
                temp / OUT_REVIEW.name
            ).baseline.sum() == 1,
            "Export preserves review-rule baseline",
        )

        for dest in (
            OUT_REVIEW,
            OUT_HOT,
            OUT_GRID,
            OUT_QA,
        ):

            if dest.exists():
                raise FileExistsError(dest)

        for _, src, dest in paths:
            src.replace(dest)

    # ========================================================
    # 8. VERIFY SOURCE INTEGRITY
    # ========================================================

    print(
        "\n[8] VERIFY SOURCE FILES WERE NOT MODIFIED"
    )

    for p in sources:

        must(
            hash_file(p) == inputs[rel(p)],
            f"Unchanged: {p.name}",
        )

    # ========================================================
    # 9. FINAL SENSITIVITY QA REPORT
    # ========================================================

    stress_r = review.loc[
        ~review.baseline
    ]

    qa_out = {

        "stage":
            "18h_combined_planning_review_and_hotspot_sensitivity",

        "status": (
            "PASS_REVIEW_AND_SPATIAL_HOTSPOT_SENSITIVITY_"
            "WITH_BASELINE_REPRODUCTION"
        ),

        "grid_cells":
            N,

        "review_rule_scenarios":
            9,

        "hotspot_scenarios_per_index":
            9,

        "review_suitability_and_cp_cutoffs_pct":
            list(RULE_PCTS),

        "hotspot_top_domain_cutoffs_pct":
            list(HOT_PCTS),

        "hotspot_min_connected_cells":
            list(MIN_SIZES),

        "shared_edge_length_threshold_m":
            EDGE_M,

        "frozen_original_review_joint_concerns":
            7,

        "frozen_original_review_feasibility_screens":
            1,

        "baseline_dci_hotspot_count":
            36,

        "baseline_conservation_hotspot_count":
            22,

        "stress_review_joint_concern_count_range": [
            int(stress_r.joint_concern_cells.min()),
            int(stress_r.joint_concern_cells.max()),
        ],

        "stress_review_managed_growth_count_range": [
            int(stress_r.managed_growth_screen_cells.min()),
            int(stress_r.managed_growth_screen_cells.max()),
        ],

        "max_changed_review_rule_cells":
            int(
                stress_r.changed_review_rule_cells.max()
            ),

        "ever_changed_common_domain_review_cells":
            int(
                np.count_nonzero(
                    change_counter[cp_valid]
                )
            ),

        "hotspot_candidate_count_range_by_indicator": {
            name: [
                int(x.candidate_hotspots.min()),
                int(x.candidate_hotspots.max()),
            ]
            for name, x in hotspots.groupby("indicator")
        },

        "source_scores_or_expert_weights_modified":
            False,

        "pressure_threshold_changed":
            False,

        "composite_dci_cp_score_created":
            False,

        "statistically_significant_hotspot_claimed":
            False,

        "official_landuse_priority_assigned":
            False,

        "limitations": [
            (
                "All alternative cutoffs and minimum sizes "
                "are exploratory sensitivity probes, "
                "not approved policy thresholds."
            ),
            (
                "Edge-sharing cluster adjacency uses "
                "polygon boundary intersections "
                "greater than 0.10 metres."
            ),
            (
                "Sensitivity is to screening choices, "
                "not uncertainty in AHP judgments "
                "or model predictions."
            ),
            (
                "DCI and CP use separate domains, and CP "
                "already incorporates pressure; their "
                "scores must not be aggregated."
            ),
            (
                "Stable screening does not replace "
                "field, zoning, drainage and flood-hazard "
                "verification."
            ),
            (
                "Spatial clustering alone does not establish "
                "statistical hotspot significance."
            ),
            (
                "GN polygon coverage gaps from 18I-01 "
                "remain unassigned and unchanged."
            ),
        ],

        "inputs_sha256":
            inputs,

        "outputs_sha256": {
            rel(p): hash_file(p)
            for p in (
                OUT_REVIEW,
                OUT_HOT,
                OUT_GRID,
            )
        },
    }

    OUT_QA.write_text(
        json.dumps(
            qa_out,
            indent=2,
        ),
        encoding="utf-8",
    )

    # ========================================================
    # 10. TERMINAL SUMMARY
    # ========================================================

    print("\n" + "=" * 78)

    print(
        qa_out["status"]
    )

    print(
        "Review: 9 scenarios; "
        "hotspot: 9 DCI + 9 CP scenarios"
    )

    print(
        "Joint-concern range:",
        qa_out[
            "stress_review_joint_concern_count_range"
        ],
    )

    print(
        "Feasibility range:",
        qa_out[
            "stress_review_managed_growth_count_range"
        ],
    )

    print(
        "DCI hotspot count range:",
        qa_out[
            "hotspot_candidate_count_range_by_indicator"
        ]["DCI"],
    )

    print(
        "CP hotspot count range:",
        qa_out[
            "hotspot_candidate_count_range_by_indicator"
        ]["CP"],
    )

    print(
        "All prior outputs preserved: YES"
    )

    print("\nReview scenarios:", OUT_REVIEW)
    print("Hotspot scenarios:", OUT_HOT)
    print("Grid stability:", OUT_GRID)
    print("QA:", OUT_QA)

    print("=" * 78)


if __name__ == "__main__":
    main()
