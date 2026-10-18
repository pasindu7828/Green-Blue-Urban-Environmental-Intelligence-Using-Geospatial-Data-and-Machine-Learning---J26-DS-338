
"""
J26-DS-338 | Section 05 | Step 18I-02
EXPLORATORY SPATIAL CONFLICT HOTSPOTS (100 m grids)

- DCI and frozen CP analysed SEPARATELY.
- Top 10% by existing validated ranks in each valid domain.
- Edge-sharing polygon adjacency; corner-only contact does not connect.
- Connected components with >=3 cells are candidate hotspots.
- Small components are kept as non-hotspot screened groups.
- GN areas use validated fractional polygon intersections.
- Unattributed GN-boundary gaps are preserved, never filled.
- NOT a statistical hotspot test or legal planning classification.
"""

from pathlib import Path
from tempfile import TemporaryDirectory
from collections import defaultdict
import hashlib
import json
import math

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely import force_2d, union_all


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

GRID_GPKG = (
    BASE / "common_grid"
    / "18b_common_grid_integration_100m.gpkg"
)

GRID_QA = (
    REPORT / "18b_common_grid_integration_qa.json"
)

QUEUE_CSV = (
    BASE / "intervention_prioritization"
    / "18g_separate_planning_review_queues_100m.csv"
)

QUEUE_QA = (
    REPORT / "18g_separate_planning_review_queues_qa.json"
)

PIECES_CSV = (
    BASE / "gn_hotspots"
    / "18i_01_gn_grid_intersections.csv"
)

ASSIGN_CSV = (
    BASE / "gn_hotspots"
    / "18i_01_grid_gn_assignment_100m.csv"
)

GN_SUMMARY_CSV = (
    BASE / "gn_hotspots"
    / "18i_01_gn_area_weighted_summary.csv"
)

GN_QA = (
    REPORT / "18i_01_gn_area_weighted_summary_qa.json"
)

OUT = BASE / "gn_hotspots"

OUT_GRID = (
    OUT
    / "18i_02_spatial_conflict_hotspot_grid_membership_100m.csv"
)

OUT_CLUSTERS = (
    OUT / "18i_02_spatial_conflict_hotspot_clusters.csv"
)

OUT_GN = (
    OUT / "18i_02_gn_hotspot_summary.csv"
)

OUT_GPKG = (
    OUT / "18i_02_spatial_conflict_hotspots.gpkg"
)

OUT_QA = (
    REPORT / "18i_02_spatial_conflict_hotspots_qa.json"
)


# ============================================================
# 2. FROZEN RESEARCH CONTRACT
# ============================================================

GRID_LAYER = "planning_conflict_common_grid_100m"

CLUSTER_LAYER = "screened_spatial_groups"

N = 9240
P_N = 4620
CP_N = 4609
GN_N = 57
P_FLAGGED = 255

TOP_FRACTION = 0.10

MIN_HOTSPOT_CELLS = 3

# Must share more than a tiny edge segment.
# Corner-only touching does not count.
MIN_SHARED_EDGE_M = 0.10

AREA_KM2 = 88.82921581108494


# ============================================================
# 3. VALIDATION HELPERS
# ============================================================

def must(test, label):

    ok = bool(test)

    print(
        f"[{'PASS' if ok else 'FAIL'}] {label}"
    )

    if not ok:
        raise ValueError(
            f"18I-02 QA failure: {label}"
        )


def digest(path):

    h = hashlib.sha256()

    with path.open("rb") as f:

        for block in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(block)

    return h.hexdigest()


def number(df, field):

    if field not in df:
        raise KeyError(
            f"Missing column: {field}"
        )

    return pd.to_numeric(
        df[field],
        errors="coerce",
    ).to_numpy(dtype=float)


def unique(df, field, expected, label):

    must(
        len(df) == expected
        and field in df
        and df[field].notna().all()
        and df[field].is_unique,
        f"{label}: {expected:,} distinct {field}",
    )


def check_range(values, domain, name):

    must(
        not np.isinf(values).any()
        and np.array_equal(
            np.isfinite(values),
            domain,
        )
        and np.all(
            (values[domain] >= -1e-12)
            & (values[domain] <= 1 + 1e-12)
        ),
        f"{name}: bounded score and exact missing-data domain",
    )


# ============================================================
# 4. EDGE-CONNECTED COMPONENT DETECTION
# ============================================================

def connected_components(selected_gdf):

    """
    Create spatially connected groups.

    Two cells connect only when they share an
    actual polygon boundary segment.

    Corner-only contact is excluded.

    Returns:
        components: lists of row positions
        edges: number of qualifying adjacency links
    """

    n = len(selected_gdf)

    parent = np.arange(
        n,
        dtype=int,
    )

    rank = np.zeros(
        n,
        dtype=int,
    )

    def find(x):

        while parent[x] != x:

            parent[x] = parent[parent[x]]

            x = parent[x]

        return x

    def union(a, b):

        ra = find(a)
        rb = find(b)

        if ra == rb:
            return

        if rank[ra] < rank[rb]:
            ra, rb = rb, ra

        parent[rb] = ra

        if rank[ra] == rank[rb]:
            rank[ra] += 1

    geometries = list(
        selected_gdf.geometry
    )

    spatial_index = selected_gdf.sindex

    edges = 0

    for i, geom in enumerate(geometries):

        candidates = spatial_index.query(
            geom,
            predicate="intersects",
        )

        for j0 in candidates:

            j = int(j0)

            if j <= i:
                continue

            shared_length = (
                geom.boundary.intersection(
                    geometries[j].boundary
                ).length
            )

            if shared_length > MIN_SHARED_EDGE_M:

                union(i, j)

                edges += 1

    groups = defaultdict(list)

    for i in range(n):

        groups[find(i)].append(i)

    components = list(
        groups.values()
    )

    # Deterministic ordering:
    # larger groups first, then grid_id.
    components.sort(
        key=lambda members: (
            -len(members),
            min(
                str(selected_gdf.iloc[k]["grid_id"])
                for k in members
            ),
        )
    )

    return components, edges


# ============================================================
# 5. ANALYZE ONE HOTSPOT INDICATOR
# ============================================================

def analyze_type(
    kind,
    prefix,
    rank_field,
    score_field,
    base,
    pieces,
):

    score = number(
        base,
        score_field,
    )

    ranks = number(
        base,
        rank_field,
    )

    domain = np.isfinite(score)

    valid_n = int(
        domain.sum()
    )

    cutoff_n = math.ceil(
        TOP_FRACTION * valid_n
    )

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
                1,
                valid_n + 1,
            ),
        ),
        f"{kind}: existing source ranks cover exact score domain",
    )

    screened = (
        domain
        & (ranks <= cutoff_n)
    )

    must(
        int(screened.sum()) == cutoff_n,
        f"{kind}: exactly {cutoff_n} top-decile screened grids",
    )

    selected = base.loc[
        screened,
        ["grid_id", "geometry"],
    ].copy()

    selected = (
        selected
        .sort_values("grid_id")
        .reset_index(drop=True)
    )

    components, edges = connected_components(
        selected
    )

    # All grids remain represented in the output.
    membership = pd.DataFrame({

        "grid_id":
            base["grid_id"].astype(str),

        f"{prefix}_screen_status":
            np.where(
                domain,
                "NOT_IN_TOP_DECILE",
                "SCORE_UNAVAILABLE",
            ),

        f"{prefix}_cluster_id":
            pd.Series(
                pd.NA,
                index=base.index,
                dtype="string",
            ),

        f"{prefix}_cluster_size_cells":
            pd.Series(
                pd.NA,
                index=base.index,
                dtype="Int64",
            ),

        f"{prefix}_candidate_hotspot":
            pd.Series(
                pd.NA,
                index=base.index,
                dtype="boolean",
            ),
    })

    membership.loc[
        domain,
        f"{prefix}_candidate_hotspot",
    ] = False

    by_id = base.set_index(
        "grid_id"
    )

    records = []

    # ========================================================
    # BUILD CLUSTER RECORDS
    # ========================================================

    for ordinal, members in enumerate(
        components,
        1,
    ):

        subset = selected.iloc[
            members
        ]

        ids = (
            subset["grid_id"]
            .astype(str)
            .tolist()
        )

        cluster_id = (
            f"{prefix.upper()}_GROUP_{ordinal:04d}"
        )

        candidate = (
            len(ids) >= MIN_HOTSPOT_CELLS
        )

        label = (
            "CANDIDATE_HOTSPOT"
            if candidate
            else "SMALL_SCREENED_GROUP"
        )

        polygons = list(
            subset.geometry
        )

        shape = union_all(
            polygons
        )

        cell_area = float(
            sum(g.area for g in polygons)
        )

        shape_area = float(
            shape.area
        )

        must(
            abs(cell_area - shape_area)
            <= max(
                1.0,
                cell_area * 1e-8,
            ),
            f"{cluster_id}: no area double-counting",
        )

        scores = (
            by_id.loc[
                ids,
                score_field,
            ]
            .to_numpy(dtype=float)
        )

        weights = np.array(
            [g.area for g in polygons],
            dtype=float,
        )

        # GN attribution uses actual intersected
        # areas from Step 18I-01, not just the
        # dominant GN of each grid.
        part = pieces.loc[
            pieces.grid_id.isin(ids)
        ]

        covered = float(
            part.intersection_area_m2.sum()
        )

        uncovered = max(
            0.0,
            shape_area - covered,
        )

        must(
            covered <= shape_area + 1.0,
            f"{cluster_id}: GN attribution does not exceed cluster area",
        )

        if len(part):

            gn_areas = (
                part.groupby(
                    ["gn_id", "gn_name"],
                    as_index=False,
                )["intersection_area_m2"]
                .sum()
                .sort_values(
                    [
                        "intersection_area_m2",
                        "gn_id",
                    ],
                    ascending=[
                        False,
                        True,
                    ],
                )
            )

            main_gn_id = str(
                gn_areas.iloc[0].gn_id
            )

            main_gn_name = str(
                gn_areas.iloc[0].gn_name
            )

            gn_names = " | ".join(
                sorted(
                    gn_areas.gn_name
                    .astype(str)
                    .unique()
                )
            )

            gn_n = len(
                gn_areas
            )

        else:

            main_gn_id = None

            main_gn_name = None

            gn_names = None

            gn_n = 0

        records.append({

            "index_type":
                kind,

            "cluster_id":
                cluster_id,

            "screen_status":
                label,

            "is_candidate_hotspot":
                bool(candidate),

            "grid_count":
                len(ids),

            "area_m2":
                shape_area,

            "area_ha":
                shape_area / 1e4,

            "area_km2":
                shape_area / 1e6,

            "mean_screening_score_area_weighted":
                float(
                    np.average(
                        scores,
                        weights=weights,
                    )
                ),

            "minimum_screening_score":
                float(scores.min()),

            "maximum_screening_score":
                float(scores.max()),

            "gn_covered_area_m2":
                covered,

            "gn_uncovered_area_m2":
                uncovered,

            "intersected_gn_division_count":
                gn_n,

            "dominant_intersected_gn_id":
                main_gn_id,

            "dominant_intersected_gn_name":
                main_gn_name,

            "intersected_gn_names":
                gn_names,

            "shared_edge_threshold_m":
                MIN_SHARED_EDGE_M,

            "screening_top_fraction":
                TOP_FRACTION,

            "minimum_hotspot_cells":
                MIN_HOTSPOT_CELLS,

            "geometry":
                shape,
        })

        chosen = membership.grid_id.isin(
            ids
        )

        membership.loc[
            chosen,
            f"{prefix}_screen_status",
        ] = label

        membership.loc[
            chosen,
            f"{prefix}_cluster_id",
        ] = cluster_id

        membership.loc[
            chosen,
            f"{prefix}_cluster_size_cells",
        ] = len(ids)

        membership.loc[
            chosen,
            f"{prefix}_candidate_hotspot",
        ] = candidate

    cluster_gdf = gpd.GeoDataFrame(
        records,
        geometry="geometry",
        crs=base.crs,
    )

    must(
        len(cluster_gdf) > 0
        and cluster_gdf.cluster_id.is_unique,
        f"{kind}: unique connected spatial groups",
    )

    must(
        int(
            cluster_gdf.grid_count.sum()
        ) == cutoff_n,
        f"{kind}: every screened grid belongs to exactly one group",
    )

    must(
        np.array_equal(
            membership[
                f"{prefix}_cluster_id"
            ].notna().to_numpy(),
            screened,
        ),
        f"{kind}: group membership exactly equals top-decile screen",
    )

    hot_ids = cluster_gdf.loc[
        cluster_gdf.is_candidate_hotspot,
        "cluster_id",
    ]

    hot_mask = (
        membership[
            f"{prefix}_cluster_id"
        ]
        .isin(hot_ids)
        .to_numpy()
    )

    must(
        not np.any(
            hot_mask & ~screened
        ),
        f"{kind}: candidate hotspots stay within screened domain",
    )

    # ========================================================
    # INDICATOR-SPECIFIC QA STATISTICS
    # ========================================================

    stats = {

        "valid_domain_cells":
            valid_n,

        "top_decile_screened_cells":
            cutoff_n,

        "screen_threshold_observed_min_score":
            float(
                score[screened].min()
            ),

        "candidate_hotspot_count":
            int(
                cluster_gdf.is_candidate_hotspot.sum()
            ),

        "candidate_hotspot_cells":
            int(
                hot_mask.sum()
            ),

        "candidate_hotspot_area_km2":
            float(
                cluster_gdf.loc[
                    cluster_gdf.is_candidate_hotspot,
                    "area_km2",
                ].sum()
            ),

        "small_screened_group_count":
            int(
                (
                    ~cluster_gdf.is_candidate_hotspot
                ).sum()
            ),

        "small_screened_group_cells":
            int(
                screened.sum()
                - hot_mask.sum()
            ),

        "shared_edge_links_in_screen":
            int(edges),

        "candidate_gn_uncovered_area_km2":
            float(
                cluster_gdf.loc[
                    cluster_gdf.is_candidate_hotspot,
                    "gn_uncovered_area_m2",
                ].sum() / 1e6
            ),
    }

    return (
        membership,
        cluster_gdf,
        stats,
    )


# ============================================================
# 6. MAIN WORKFLOW
# ============================================================

def main():

    print("=" * 79)

    print(
        "SECTION 05 | STEP 18I-02 | "
        "EXPLORATORY SPATIAL CONFLICT HOTSPOTS"
    )

    print("=" * 79)

    sources = [
        GRID_GPKG,
        GRID_QA,
        QUEUE_CSV,
        QUEUE_QA,
        PIECES_CSV,
        ASSIGN_CSV,
        GN_SUMMARY_CSV,
        GN_QA,
    ]

    outputs = [
        OUT_GRID,
        OUT_CLUSTERS,
        OUT_GN,
        OUT_GPKG,
        OUT_QA,
    ]

    for path in sources:

        if not path.is_file():
            raise FileNotFoundError(
                path
            )

    for path in outputs:

        if path.exists():
            raise FileExistsError(
                f"Refusing to overwrite: {path}"
            )

    hashes = {
        p.relative_to(ROOT).as_posix(): digest(p)
        for p in sources
    }

    # ========================================================
    # STEP 1 - VERIFY FROZEN SOURCE RELEASES
    # ========================================================

    print(
        "\n[1] VERIFY FROZEN RESEARCH RELEASES"
    )

    qb = json.loads(
        GRID_QA.read_text(
            encoding="utf-8"
        )
    )

    qg = json.loads(
        QUEUE_QA.read_text(
            encoding="utf-8"
        )
    )

    qn = json.loads(
        GN_QA.read_text(
            encoding="utf-8"
        )
    )

    must(
        qb.get("status")
        == "PASS_COMMON_GRID_FINAL_EXPERT_AHP_INTEGRATION",
        "18B final expert-AHP grid integration passed",
    )

    must(
        qg.get("status")
        == "PASS_SEPARATE_REVIEW_QUEUE_RANKINGS_SENSITIVITY_PENDING",
        "18G separate ranking queues passed",
    )

    must(
        qn.get("status")
        == "PASS_GN_AREA_WEIGHTED_INTEGRATION_WITH_COVERAGE_AUDIT",
        "18I-01 fractional GN integration passed",
    )

    must(
        qb["outputs"]["gpkg_sha256"]
        == digest(GRID_GPKG),
        "18B grid geometry hash verified",
    )

    must(
        qg["output"]["sha256"]
        == digest(QUEUE_CSV),
        "18G queue CSV hash verified",
    )

    for path in [
        PIECES_CSV,
        ASSIGN_CSV,
        GN_SUMMARY_CSV,
    ]:

        must(
            qn["outputs"].get(
                path.relative_to(ROOT).as_posix()
            ) == digest(path),
            f"18I-01 frozen source hash: {path.name}",
        )

    must(
        abs(
            qn["grid_area_uncovered_by_gn_km2"]
            - 2.133336
        ) < 0.001,
        "Known GN coverage gap preserved, not imputed",
    )

    # ========================================================
    # STEP 2 - LOAD GIS AND SOURCE SCORES
    # ========================================================

    print(
        "\n[2] LOAD AND VALIDATE COMMON 100-M GRID"
    )

    grid = gpd.read_file(
        GRID_GPKG,
        layer=GRID_LAYER,
    )

    queue = pd.read_csv(
        QUEUE_CSV
    )

    pieces = pd.read_csv(
        PIECES_CSV,
        dtype={"gn_id": "string"},
    )

    assign = pd.read_csv(
        ASSIGN_CSV,
        dtype={"gn_id": "string"},
    )

    gns = pd.read_csv(
        GN_SUMMARY_CSV,
        dtype={"gn_id": "string"},
    )

    unique(
        grid,
        "grid_id",
        N,
        "Final 18B grid",
    )

    unique(
        queue,
        "grid_id",
        N,
        "18G review queue",
    )

    unique(
        assign,
        "grid_id",
        N,
        "GN grid assignment",
    )

    unique(
        gns,
        "gn_id",
        GN_N,
        "GN summaries",
    )

    must(
        grid.crs is not None
        and grid.crs.to_epsg() == 32644,
        "Grid EPSG:32644",
    )

    must(
        grid.geometry.notna().all()
        and grid.geometry.is_valid.all()
        and not grid.geometry.is_empty.any(),
        "Valid grid geometries",
    )

    must(
        set(grid.grid_id)
        == set(queue.grid_id)
        == set(assign.grid_id),
        "Grid IDs identical across integration and review queues",
    )

    grid["geometry"] = [
        force_2d(g)
        for g in grid.geometry
    ]

    queue = (
        queue.set_index("grid_id")
        .loc[grid.grid_id]
        .reset_index()
    )

    assign = (
        assign.set_index("grid_id")
        .loc[grid.grid_id]
        .reset_index()
    )

    must(
        all(
            field in pieces.columns
            for field in [
                "grid_id",
                "gn_id",
                "gn_name",
                "intersection_area_m2",
            ]
        ),
        "GN fractional-intersection columns present",
    )

    must(
        pieces[
            ["grid_id", "gn_id"]
        ].notna().all().all()
        and not pieces.duplicated(
            ["grid_id", "gn_id"]
        ).any(),
        "GN-grid pairs unique and attributable",
    )

    must(
        pieces.grid_id.isin(
            grid.grid_id
        ).all()
        and pieces.gn_id.isin(
            gns.gn_id
        ).all(),
        "GN-grid intersections use real grid/GN IDs",
    )

    must(
        np.isfinite(
            number(
                pieces,
                "intersection_area_m2",
            )
        ).all()
        and (
            number(
                pieces,
                "intersection_area_m2",
            ) > 0
        ).all(),
        "Fractional GN intersection areas positive",
    )

    must(
        np.isclose(
            pieces.intersection_area_m2.sum() / 1e6,
            qn["grid_area_covered_by_gn_km2"],
            atol=1e-5,
            rtol=0,
        ),
        "Fractional GN area reconciles with 18I-01 QA",
    )

    must(
        np.isclose(
            grid.geometry.area.sum() / 1e6,
            AREA_KM2,
            atol=0.001,
            rtol=0,
        ),
        "KMC grid total area preserved",
    )

    # ========================================================
    # STEP 3 - VERIFY FINAL INDICES
    # ========================================================

    for field, other in [
        (
            "development_suitability_score",
            "development_suitability_score",
        ),
        (
            "urbanization_pressure_score",
            "urbanization_pressure_score",
        ),
    ]:

        must(
            np.allclose(
                number(grid, field),
                number(queue, other),
                equal_nan=True,
                atol=1e-12,
                rtol=0,
            ),
            f"Frozen score agrees across 18B/18G: {field}",
        )

    pressure = number(
        queue,
        "urbanization_pressure_score",
    )

    dci = number(
        queue,
        "development_conflict_index_0_1",
    )

    cp = number(
        queue,
        "frozen_conservation_priority_0_1",
    )

    suit = number(
        queue,
        "development_suitability_score",
    )

    flags = number(
        queue,
        "pressure_threshold_flag_frozen",
    )

    p_domain = np.isfinite(
        pressure
    )

    cp_domain = np.isfinite(
        cp
    )

    must(
        p_domain.sum() == P_N
        and cp_domain.sum() == CP_N,
        "4,620 pressure and 4,609 conservation-score cells",
    )

    check_range(
        dci,
        p_domain,
        "Frozen DCI",
    )

    check_range(
        cp,
        cp_domain,
        "Frozen Conservation Priority",
    )

    must(
        np.isfinite(suit).all()
        and np.allclose(
            dci[p_domain],
            pressure[p_domain]
            * (1 - suit[p_domain]),
            atol=1e-12,
            rtol=0,
        ),
        "DCI is unchanged expert-AHP suitability shortfall times pressure",
    )

    must(
        np.all(
            ~cp_domain | p_domain
        ),
        "Frozen CP restricted to pressure domain",
    )

    must(
        np.isnan(
            flags[~p_domain]
        ).all()
        and int(
            np.count_nonzero(
                flags[p_domain] == 1
            )
        ) == P_FLAGGED,
        "255 original pressure flags preserved",
    )

    base = grid[
        ["grid_id", "geometry"]
    ].copy()

    for col in [
        "development_conflict_index_0_1",
        "frozen_conservation_priority_0_1",
        "dci_rank_in_pressure_domain",
        "cp_rank_in_conservation_domain",
    ]:

        base[col] = (
            queue[col].to_numpy()
        )

    # ========================================================
    # STEP 4 - SEPARATE DCI AND CP SPATIAL CLUSTERS
    # ========================================================

    print(
        "\n[3] BUILD SEPARATE EDGE-CONNECTED SPATIAL GROUPS"
    )

    (
        dci_m,
        dci_groups,
        dci_stats,
    ) = analyze_type(

        "DEVELOPMENT_CONFLICT",

        "dci",

        "dci_rank_in_pressure_domain",

        "development_conflict_index_0_1",

        base,

        pieces,
    )

    (
        cp_m,
        cp_groups,
        cp_stats,
    ) = analyze_type(

        "CONSERVATION_PRIORITY",

        "cp",

        "cp_rank_in_conservation_domain",

        "frozen_conservation_priority_0_1",

        base,

        pieces,
    )

    clusters = gpd.GeoDataFrame(
        pd.concat(
            [
                dci_groups,
                cp_groups,
            ],
            ignore_index=True,
        ),
        geometry="geometry",
        crs=base.crs,
    )

    must(
        clusters.cluster_id.is_unique
        and clusters.geometry.is_valid.all(),
        "All clusters uniquely identified and geometrically valid",
    )

    # ========================================================
    # STEP 5 - GRID-LEVEL HOTSPOT MEMBERSHIP
    # ========================================================

    print(
        "\n[4] BUILD GRID-LEVEL MEMBERSHIP AND GN SUMMARIES"
    )

    membership = queue[
        [
            "grid_id",
            "review_rule_id",
            "pressure_domain_status",
            "urbanization_pressure_score",
            "development_suitability_score",
            "development_conflict_index_0_1",
            "frozen_conservation_priority_0_1",
        ]
    ].copy()

    membership = membership.merge(
        assign[
            [
                "grid_id",
                "gn_id",
                "gn_name",
                "gn_assignment_status",
                "gn_covered_fraction",
                "gn_uncovered_area_m2",
            ]
        ],
        on="grid_id",
        how="left",
        validate="one_to_one",
    )

    membership = membership.merge(
        dci_m,
        on="grid_id",
        validate="one_to_one",
    )

    membership = membership.merge(
        cp_m,
        on="grid_id",
        validate="one_to_one",
    )

    unique(
        membership,
        "grid_id",
        N,
        "Hotspot grid membership",
    )

    must(
        membership.loc[
            assign.gn_id.isna(),
            "gn_name",
        ].isna().all(),
        "Grids without GN coverage remain unnamed",
    )

    # ========================================================
    # STEP 6 - GN-LEVEL HOTSPOT SUMMARIES
    # ========================================================

    gn_results = gns[
        [
            "gn_id",
            "gn_name",
            "gn_area_inside_kmc_km2",
            "mapped_grid_area_km2",
        ]
    ].copy()

    for prefix in (
        "dci",
        "cp",
    ):

        status_col = (
            f"{prefix}_screen_status"
        )

        cluster_col = (
            f"{prefix}_cluster_id"
        )

        selected = membership.loc[
            membership[status_col].eq(
                "CANDIDATE_HOTSPOT"
            ),
            [
                "grid_id",
                cluster_col,
            ],
        ].copy()

        joined = pieces.merge(
            selected,
            on="grid_id",
            how="inner",
            validate="many_to_one",
        )

        if len(joined):

            aggregates = joined.groupby(
                "gn_id",
                as_index=False,
            ).agg(
                **{
                    f"{prefix}_hotspot_gn_area_m2": (
                        "intersection_area_m2",
                        "sum",
                    ),
                    f"{prefix}_hotspot_intersecting_grid_count": (
                        "grid_id",
                        "nunique",
                    ),
                    f"{prefix}_hotspot_cluster_count": (
                        cluster_col,
                        "nunique",
                    ),
                }
            )

        else:

            aggregates = pd.DataFrame(
                columns=[
                    "gn_id",
                    f"{prefix}_hotspot_gn_area_m2",
                    f"{prefix}_hotspot_intersecting_grid_count",
                    f"{prefix}_hotspot_cluster_count",
                ]
            )

        gn_results = gn_results.merge(
            aggregates,
            on="gn_id",
            how="left",
            validate="one_to_one",
        )

        for field in [
            f"{prefix}_hotspot_gn_area_m2",
            f"{prefix}_hotspot_intersecting_grid_count",
            f"{prefix}_hotspot_cluster_count",
        ]:

            gn_results[field] = (
                gn_results[field].fillna(0)
            )

        gn_results[
            f"{prefix}_hotspot_gn_area_ha"
        ] = (
            gn_results[
                f"{prefix}_hotspot_gn_area_m2"
            ] / 1e4
        )

        gn_results[
            f"{prefix}_hotspot_share_of_gn_pct"
        ] = (
            100
            * gn_results[
                f"{prefix}_hotspot_gn_area_m2"
            ]
            / (
                gn_results.gn_area_inside_kmc_km2
                * 1e6
            )
        )

        must(
            np.all(
                gn_results[
                    f"{prefix}_hotspot_share_of_gn_pct"
                ] <= 100.1
            ),
            f"{prefix.upper()}: GN attributed hotspot area not overstated",
        )

    unique(
        gn_results,
        "gn_id",
        GN_N,
        "GN hotspot summary",
    )

    # ========================================================
    # STEP 7 - RECONCILE HOTSPOT / GN AREAS
    # ========================================================

    for prefix, st in [
        ("dci", dci_stats),
        ("cp", cp_stats),
    ]:

        actual = float(
            gn_results[
                f"{prefix}_hotspot_gn_area_m2"
            ].sum()
        )

        index_type = (
            "DEVELOPMENT_CONFLICT"
            if prefix == "dci"
            else "CONSERVATION_PRIORITY"
        )

        target = float(
            clusters.loc[
                (
                    clusters.index_type == index_type
                )
                & clusters.is_candidate_hotspot,
                "gn_covered_area_m2",
            ].sum()
        )

        must(
            abs(actual - target) <= 0.05,
            f"{prefix.upper()}: hotspot GN-intersection areas reconcile to clusters",
        )

        total = (
            st["candidate_hotspot_area_km2"]
            * 1e6
        )

        gap = (
            st["candidate_gn_uncovered_area_km2"]
            * 1e6
        )

        must(
            abs(
                (actual + gap) - total
            ) <= max(
                1.0,
                total * 1e-8,
            ),
            f"{prefix.upper()}: hotspot area equals attributed plus GN-gap area",
        )

    # ========================================================
    # STEP 8 - EXPORT GIS AND CSV DELIVERABLES
    # ========================================================

    print(
        "\n[5] EXPORT FIVE AUDITABLE DELIVERABLES"
    )

    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORT.mkdir(
        parents=True,
        exist_ok=True,
    )

    with TemporaryDirectory(
        prefix="18i02_build_",
        dir=OUT,
    ) as temp:

        temp = Path(temp)

        p_grid = (
            temp / OUT_GRID.name
        )

        p_clusters = (
            temp / OUT_CLUSTERS.name
        )

        p_gn = (
            temp / OUT_GN.name
        )

        p_gpkg = (
            temp / OUT_GPKG.name
        )

        membership.to_csv(
            p_grid,
            index=False,
        )

        clusters.drop(
            columns="geometry"
        ).to_csv(
            p_clusters,
            index=False,
        )

        gn_results.to_csv(
            p_gn,
            index=False,
        )

        clusters.to_file(
            p_gpkg,
            layer=CLUSTER_LAYER,
            driver="GPKG",
            engine="pyogrio",
        )

        check_grid = pd.read_csv(
            p_grid
        )

        check_clusters = pd.read_csv(
            p_clusters
        )

        check_gn = pd.read_csv(
            p_gn,
            dtype={"gn_id": "string"},
        )

        check_gis = gpd.read_file(
            p_gpkg,
            layer=CLUSTER_LAYER,
        )

        unique(
            check_grid,
            "grid_id",
            N,
            "Exported hotspot grid",
        )

        unique(
            check_clusters,
            "cluster_id",
            len(clusters),
            "Exported clusters",
        )

        unique(
            check_gn,
            "gn_id",
            GN_N,
            "Exported GN hotspot summary",
        )

        unique(
            check_gis,
            "cluster_id",
            len(clusters),
            "Exported hotspot GIS",
        )

        must(
            check_gis.crs is not None
            and check_gis.crs.to_epsg() == 32644
            and check_gis.geometry.is_valid.all(),
            "Exported hotspot polygons valid in EPSG:32644",
        )

        must(
            check_grid.dci_screen_status.equals(
                membership.dci_screen_status
            )
            and check_grid.cp_screen_status.equals(
                membership.cp_screen_status
            ),
            "Exported grid-level screening categories unchanged",
        )

        must(
            np.allclose(
                number(
                    check_clusters,
                    "area_m2",
                ),
                number(
                    clusters,
                    "area_m2",
                ),
                atol=1e-6,
                rtol=0,
            ),
            "Exported cluster areas unchanged",
        )

        must(
            np.allclose(
                number(
                    check_gis,
                    "area_m2",
                ),
                number(
                    clusters,
                    "area_m2",
                ),
                atol=1e-6,
                rtol=0,
            ),
            "Exported GIS cluster areas unchanged",
        )

        for dest in outputs:

            if dest.exists():
                raise FileExistsError(
                    f"Output appeared during run: {dest}"
                )

        for src, dest in [
            (p_grid, OUT_GRID),
            (p_clusters, OUT_CLUSTERS),
            (p_gn, OUT_GN),
            (p_gpkg, OUT_GPKG),
        ]:

            src.replace(
                dest
            )

    # ========================================================
    # STEP 9 - VERIFY INPUT SOURCE INTEGRITY
    # ========================================================

    print(
        "\n[6] VERIFY ALL FROZEN SOURCES UNCHANGED"
    )

    for path in sources:

        key = (
            path.relative_to(ROOT)
            .as_posix()
        )

        must(
            digest(path) == hashes[key],
            f"Source unchanged: {path.name}",
        )

    # ========================================================
    # STEP 10 - GENERATE QA REPORT
    # ========================================================

    qa = {

        "section":
            "05_planning_conflict_intervention",

        "stage":
            "18i_02_spatial_conflict_hotspots",

        "status": (
            "PASS_EXPLORATORY_DCI_CP_SPATIAL_HOTSPOTS_"
            "WITH_GN_GAPS"
        ),

        "grid_count":
            N,

        "gn_division_count":
            GN_N,

        "crs":
            "EPSG:32644",

        "top_fraction":
            TOP_FRACTION,

        "minimum_candidate_hotspot_cells":
            MIN_HOTSPOT_CELLS,

        "edge_adjacency": (
            f"Shared boundary length > "
            f"{MIN_SHARED_EDGE_M} metres "
            f"(not corner-only)"
        ),

        "dci":
            dci_stats,

        "conservation_priority":
            cp_stats,

        "gn_boundary_gap_km2":
            float(
                qn[
                    "grid_area_uncovered_by_gn_km2"
                ]
            ),

        "scores_retrained_or_changed":
            False,

        "composite_dci_cp_risk_score_created":
            False,

        "hotspots_statistically_significant":
            False,

        "legal_or_zoning_classifications_issued":
            False,

        "limitations": [

            (
                "Top 10% is a descriptive threshold; "
                "it must be tested in Step 5.9."
            ),

            (
                "Minimum three edge-connected grids is "
                "a descriptive cluster-size rule, "
                "not an official threshold."
            ),

            (
                "Screened one- or two-cell components "
                "are retained as small groups, "
                "not candidate hotspots."
            ),

            (
                "Edge adjacency uses real clipped "
                "polygon boundaries and excludes "
                "corner-only contact."
            ),

            (
                "DCI and frozen CP use separate ranking "
                "queues, but are not statistically "
                "independent signals: CP already "
                "includes pressure."
            ),

            (
                "Candidate hotspots are not statistically "
                "significant hotspots without a "
                "spatial significance test."
            ),

            (
                "Grid portions outside GN coverage "
                "remain unattributed, never "
                "nearest-filled."
            ),

            (
                "Out-of-domain and missing score cells "
                "are not treated as zero conflict "
                "or zero conservation value."
            ),

            (
                "No new legal development suitability, "
                "flood risk, approvals or official "
                "conservation boundaries are inferred."
            ),
        ],

        "input_sha256":
            hashes,

        "outputs": {

            p.relative_to(ROOT).as_posix():
                digest(p)

            for p in [
                OUT_GRID,
                OUT_CLUSTERS,
                OUT_GN,
                OUT_GPKG,
            ]
        },

        "gpkg_layer":
            CLUSTER_LAYER,
    }

    OUT_QA.write_text(
        json.dumps(
            qa,
            indent=2,
        ),
        encoding="utf-8",
    )

    # ========================================================
    # STEP 11 - TERMINAL RESULTS
    # ========================================================

    print(
        "\n" + "=" * 79
    )

    print(
        qa["status"]
    )

    for name, st in [
        ("DCI", dci_stats),
        ("CONSERVATION", cp_stats),
    ]:

        print(
            f"{name} screened:",
            st["top_decile_screened_cells"],
        )

        print(
            f"{name} candidate hotspots:",
            st["candidate_hotspot_count"],
        )

        print(
            f"{name} candidate hotspot cells:",
            st["candidate_hotspot_cells"],
        )

        print(
            f"{name} candidate hotspot area:",
            f"{st['candidate_hotspot_area_km2']:.6f} km2",
        )

        print(
            f"{name} small screened groups:",
            st["small_screened_group_count"],
        )

    print(
        "Unattributed GN boundary area remains:",
        f"{qa['gn_boundary_gap_km2']:.6f} km2",
    )

    print(
        "Official classifications / composite index created: NO"
    )

    print("\nGrid membership:", OUT_GRID)
    print("Cluster summaries:", OUT_CLUSTERS)
    print("GN hotspot summary:", OUT_GN)
    print("Cluster GIS:", OUT_GPKG)
    print("QA report:", OUT_QA)

    print(
        "=" * 79
    )


if __name__ == "__main__":
    main()
