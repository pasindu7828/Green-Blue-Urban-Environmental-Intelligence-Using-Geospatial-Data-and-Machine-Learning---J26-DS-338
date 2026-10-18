
"""
J26-DS-338 | Section 04 - Conservation Intelligence
Stage 4.5: Green Connectivity Candidate Indicators

Source: Frozen 2025 RAW_RF current-green baseline.

Candidate measurements:
  1. Neighbourhood green context: 250 m and 500 m.
  2. Four-neighbour connected majority-green grid patches.
  3. Patch area and number of cells.

NEIGHBOURHOOD METHOD:
- Use full-grid cell centres and centre-to-centre distances.
- Exclude the focal cell from neighbourhood context.
- Weight observed green by valid mapped LULC area.
- Retain neighbourhood LULC coverage as QA.
- This is a grid-centre neighbourhood approximation,
  not exact fractional clipping to circular buffers.

PATCH METHOD:
- Exploratory 100 m grid-topology measure.
- A patch-candidate cell has >50% of its actual
  KMC-intersecting area confirmed mapped green.
- Require four-direction grid adjacency AND an actual
  shared clipped geometry edge.
- Patch areas sum confirmed mapped green area.
- These are coarse grid patches, not verified
  ecological habitat corridors.

The 50% rule is NOT a final conservation candidate mask.

Do not use pressure, SHAP, suitability, AHP,
historical threat, future threat or zoning.

No Green Importance or Conservation Priority score.

Outputs:
  17d_green_connectivity_2025_100m.csv
  17d_green_connectivity_2025_100m.gpkg
  17d_green_connectivity_qa.json
"""

from pathlib import Path
import json

import geopandas as gpd
import numpy as np
import pandas as pd
import pyogrio

from scipy.spatial import cKDTree


ROOT = Path(__file__).resolve().parents[2]

GRID_FILE = (
    ROOT / "data/processed/urbanization_pressure/grid/"
    "kaduwela_100m_modelling_grid.gpkg"
)

CURRENT_DIR = (
    ROOT / "data/processed/conservation_intelligence/"
    "current_green"
)

CURRENT_CSV = (
    CURRENT_DIR / "17b_current_green_condition_2025_100m.csv"
)

CURRENT_GPKG = (
    CURRENT_DIR / "17b_current_green_condition_2025_100m.gpkg"
)

CURRENT_QA = (
    ROOT / "outputs/reports/conservation_intelligence/"
    "17b_current_green_condition_qa.json"
)

OUTPUT_DIR = (
    ROOT / "data/processed/conservation_intelligence/"
    "connectivity"
)

REPORT_DIR = (
    ROOT / "outputs/reports/conservation_intelligence"
)

CSV_OUT = (
    OUTPUT_DIR / "17d_green_connectivity_2025_100m.csv"
)

GPKG_OUT = (
    OUTPUT_DIR / "17d_green_connectivity_2025_100m.gpkg"
)

QA_OUT = (
    REPORT_DIR / "17d_green_connectivity_qa.json"
)

RADII = [250, 500]
PATCH_MAJORITY_THRESHOLD = 0.50


def require_columns(frame, columns, label):

    missing = sorted(
        set(columns) - set(frame.columns)
    )

    if missing:
        raise ValueError(
            f"{label}: missing columns {missing}"
        )


def distribution(values):

    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if len(values) == 0:
        return {"count": 0}

    return {
        "count": int(len(values)),
        "min": float(values.min()),
        "p05": float(np.quantile(values, 0.05)),
        "p25": float(np.quantile(values, 0.25)),
        "median": float(np.median(values)),
        "p75": float(np.quantile(values, 0.75)),
        "p95": float(np.quantile(values, 0.95)),
        "max": float(values.max()),
    }


def calculate_neighbourhood(
    centres,
    kmc_areas,
    valid_areas,
    green_areas,
    radius,
):
    """
    Grid-centre distance neighbourhood.

    Only neighbours are included; focal cell excluded.
    Known green area / valid mapped area is the
    observed green context.

    Unobserved LULC area is not treated as non-green.
    """

    tree = cKDTree(centres)

    neighbour_lists = tree.query_ball_point(
        centres, r=radius
    )

    n = len(centres)

    green_context = np.full(
        n, np.nan, dtype=float
    )

    mapped_coverage = np.full(
        n, np.nan, dtype=float
    )

    counts = np.zeros(n, dtype=int)
    observed_counts = np.zeros(n, dtype=int)

    neighbourhood_green_area = np.zeros(
        n, dtype=float
    )

    neighbourhood_valid_area = np.zeros(
        n, dtype=float
    )

    neighbourhood_kmc_area = np.zeros(
        n, dtype=float
    )

    for i, neighbours in enumerate(neighbour_lists):

        idx = np.asarray(
            neighbours, dtype=int
        )

        # Exclude focal cell so context does not merely
        # repeat the cell's own green fraction.
        idx = idx[idx != i]

        counts[i] = len(idx)

        if len(idx) == 0:
            continue

        total_kmc = float(kmc_areas[idx].sum())
        total_valid = float(valid_areas[idx].sum())
        total_green = float(green_areas[idx].sum())

        observed_counts[i] = int(
            np.count_nonzero(valid_areas[idx] > 1e-9)
        )

        neighbourhood_kmc_area[i] = total_kmc
        neighbourhood_valid_area[i] = total_valid
        neighbourhood_green_area[i] = total_green

        if total_kmc > 0:
            mapped_coverage[i] = (
                total_valid / total_kmc
            )

        if total_valid > 1e-9:
            green_context[i] = (
                total_green / total_valid
            )

    return {
        "context": green_context,
        "coverage": mapped_coverage,
        "counts": counts,
        "observed_counts": observed_counts,
        "green_area": neighbourhood_green_area,
        "valid_area": neighbourhood_valid_area,
        "kmc_area": neighbourhood_kmc_area,
    }


class UnionFind:

    def __init__(self, n):
        self.parent = list(range(n))
        self.size = [1] * n

    def find(self, x):

        while self.parent[x] != x:
            self.parent[x] = self.parent[
                self.parent[x]
            ]
            x = self.parent[x]

        return x

    def union(self, a, b):

        root_a = self.find(a)
        root_b = self.find(b)

        if root_a == root_b:
            return

        if self.size[root_a] < self.size[root_b]:
            root_a, root_b = root_b, root_a

        self.parent[root_b] = root_a
        self.size[root_a] += self.size[root_b]


def calculate_patches(
    rows,
    cols,
    confirmed_green_share,
    green_areas,
    geometries,
):
    """
    Candidate four-neighbour green-patch structure.

    Four-neighbour means cell sides, not diagonals.
    Real clipped geometries must share an edge.

    This is a coarse 100 m topology proxy.
    """

    n = len(rows)

    candidate = (
        confirmed_green_share >
        PATCH_MAJORITY_THRESHOLD
    )

    positions = {}

    for i in range(n):
        key = (int(rows[i]), int(cols[i]))

        if key in positions:
            raise ValueError(
                "Duplicate grid row/column coordinate"
            )

        positions[key] = i

    uf = UnionFind(n)

    shared_edge_checks = 0
    accepted_edges = 0

    for i in np.flatnonzero(candidate):

        row = int(rows[i])
        col = int(cols[i])

        # Right and down only; undirected graph edges
        # are therefore checked once.
        neighbours = [
            (row, col + 1),
            (row + 1, col),
        ]

        for key in neighbours:

            j = positions.get(key)

            if j is None or not candidate[j]:
                continue

            shared_edge_checks += 1

            edge_length = (
                geometries.iloc[i].boundary
                .intersection(
                    geometries.iloc[j].boundary
                )
                .length
            )

            # Reject mere point contact or geometrically
            # disconnected KMC boundary fragments.
            if edge_length > 1e-6:
                uf.union(i, j)
                accepted_edges += 1

    components = {}

    for i in np.flatnonzero(candidate):

        root = uf.find(int(i))

        components.setdefault(root, []).append(
            int(i)
        )

    # Deterministic patch numbering.
    ordered_components = sorted(
        components.values(),
        key=lambda members: min(members),
    )

    patch_ids = np.full(
        n, None, dtype=object
    )

    patch_cell_count = np.full(
        n, np.nan, dtype=float
    )

    patch_green_area_km2 = np.full(
        n, np.nan, dtype=float
    )

    patch_count = len(ordered_components)

    patch_areas = []

    for number, members in enumerate(
        ordered_components, start=1
    ):

        patch_id = f"GP4_{number:05d}"

        members_array = np.asarray(
            members, dtype=int
        )

        area_km2 = float(
            green_areas[members_array].sum()
            / 1e6
        )

        patch_areas.append(area_km2)

        patch_ids[members_array] = patch_id

        patch_cell_count[members_array] = (
            len(members)
        )

        patch_green_area_km2[members_array] = (
            area_km2
        )

    return {
        "candidate": candidate,
        "patch_ids": patch_ids,
        "patch_cell_count": patch_cell_count,
        "patch_green_area_km2": patch_green_area_km2,
        "patch_count": patch_count,
        "patch_areas": patch_areas,
        "shared_edge_checks": shared_edge_checks,
        "accepted_edges": accepted_edges,
    }


def main():

    print("=" * 72)
    print("SECTION 04 | STAGE 4.5 - GREEN CONNECTIVITY")
    print("=" * 72)

    for path in [
        GRID_FILE,
        CURRENT_CSV,
        CURRENT_GPKG,
        CURRENT_QA,
    ]:
        if not path.is_file():
            raise FileNotFoundError(path)

    audit = json.loads(
        CURRENT_QA.read_text(encoding="utf-8")
    )

    if audit.get("status") != (
        "PASS_CURRENT_GREEN_BASELINE"
    ):
        raise ValueError(
            "Stage 4.3 current green is not approved for use"
        )

    checks = {}

    def check(name, condition):

        passed = bool(condition)
        checks[name] = passed

        print(
            f"[{'PASS' if passed else 'FAIL'}] {name}"
        )

    # ==================================================
    # 1. READ AND ALIGN CURRENT GREEN DATA
    # ==================================================

    print("\n[1] READ CURRENT GREEN AND GRID DATA")

    current = pd.read_csv(CURRENT_CSV)
    raw_grid = gpd.read_file(GRID_FILE)
    clipped = gpd.read_file(CURRENT_GPKG)

    required_current = [
        "grid_id",
        "kmc_overlap_area_m2",
        "valid_lulc_area_m2",
        "current_green_area_m2",
        "current_green_fraction",
        "valid_lulc_coverage_fraction",
        "mapped_green_share_lower_bound",
        "green_condition_status",
        "source_product_type",
    ]

    required_grid = [
        "grid_id",
        "grid_row",
        "grid_col",
        "centroid_x",
        "centroid_y",
    ]

    require_columns(
        current,
        required_current,
        "Current green",
    )

    require_columns(
        raw_grid,
        required_grid,
        "Modelling grid",
    )

    check(
        "current_9240_unique_ids",
        len(current) == 9240
        and current.grid_id.notna().all()
        and current.grid_id.is_unique,
    )

    check(
        "grid_gis_9240_unique_ids",
        len(raw_grid) == 9240
        and raw_grid.grid_id.notna().all()
        and raw_grid.grid_id.is_unique,
    )

    check(
        "clipped_gis_9240_unique_ids",
        len(clipped) == 9240
        and clipped.grid_id.notna().all()
        and clipped.grid_id.is_unique,
    )

    check(
        "all_grid_id_sets_match",
        set(current.grid_id)
        == set(raw_grid.grid_id)
        == set(clipped.grid_id),
    )

    check(
        "all_gis_epsg32644",
        raw_grid.crs is not None
        and clipped.crs is not None
        and raw_grid.crs.to_epsg() == 32644
        and clipped.crs.to_epsg() == 32644,
    )

    check(
        "2025_raw_rf_source",
        current.source_product_type.eq("RAW_RF").all(),
    )

    if not all(checks.values()):
        raise SystemExit(
            "Source QA failed. No output exported."
        )

    ids = current.grid_id.tolist()

    raw_grid = (
        raw_grid.set_index("grid_id")
        .loc[ids]
    )

    clipped = (
        clipped.set_index("grid_id")
        .loc[ids]
    )

    centres = raw_grid[
        ["centroid_x", "centroid_y"]
    ].to_numpy(dtype=float)

    rows = raw_grid.grid_row.to_numpy(dtype=int)
    cols = raw_grid.grid_col.to_numpy(dtype=int)

    areas = current[
        "kmc_overlap_area_m2"
    ].to_numpy(dtype=float)

    valid_areas = current[
        "valid_lulc_area_m2"
    ].to_numpy(dtype=float)

    green_areas = current[
        "current_green_area_m2"
    ].to_numpy(dtype=float)

    observed_green = current[
        "current_green_fraction"
    ].to_numpy(dtype=float)

    confirmed_share = current[
        "mapped_green_share_lower_bound"
    ].to_numpy(dtype=float)

    observed = valid_areas > 1e-9

    check(
        "centres_finite",
        np.isfinite(centres).all(),
    )

    check(
        "areas_valid",
        np.isfinite(areas).all()
        and np.isfinite(valid_areas).all()
        and np.isfinite(green_areas).all()
        and (areas > 0).all()
        and (valid_areas >= 0).all()
        and (green_areas >= 0).all()
        and (valid_areas <= areas + 1.0).all()
        and (green_areas <= valid_areas + 1e-6).all(),
    )

    check(
        "unknown_green_preserved",
        np.isnan(observed_green[~observed]).all()
        and np.isfinite(
            observed_green[observed]
        ).all(),
    )

    check(
        "confirmed_green_share_valid",
        np.isfinite(confirmed_share).all()
        and (
            (confirmed_share >= 0)
            & (confirmed_share <= 1)
        ).all()
        and np.allclose(
            confirmed_share,
            green_areas / areas,
            atol=1e-6,
        ),
    )

    geometry_area_error = float(
        np.max(
            np.abs(
                clipped.geometry.area.to_numpy(dtype=float)
                - areas
            )
        )
    )

    check(
        "clipped_geometry_valid_and_area_matched",
        clipped.geometry.notna().all()
        and clipped.geometry.is_valid.all()
        and geometry_area_error <= 1.0,
    )

    if not all(checks.values()):
        raise SystemExit(
            "Spatial or green-source QA failed."
        )

    print("Grid cells:", len(current))
    print("Current green source: 2025 RAW_RF")
    print("Unknown current-green cells:", int((~observed).sum()))

    # ==================================================
    # 2. GREEN NEIGHBOURHOOD CONTEXT
    # ==================================================

    print("\n[2] CALCULATE GREEN NEIGHBOURHOOD CONTEXT")

    contexts = {}

    for radius in RADII:

        context = calculate_neighbourhood(
            centres=centres,
            kmc_areas=areas,
            valid_areas=valid_areas,
            green_areas=green_areas,
            radius=radius,
        )

        contexts[radius] = context

        available = np.isfinite(
            context["context"]
        )

        print(
            f"{radius} m | "
            f"Valid neighbourhoods: {int(available.sum()):,} | "
            f"No observed neighbours: {int((~available).sum()):,}"
        )

        print(
            "  Green context:",
            distribution(context["context"]),
        )

    # ==================================================
    # 3. EXPLORATORY GREEN-PATCH STRUCTURE
    # ==================================================

    print("\n[3] CALCULATE GREEN PATCH STRUCTURE")

    patches = calculate_patches(
        rows=rows,
        cols=cols,
        confirmed_green_share=confirmed_share,
        green_areas=green_areas,
        geometries=clipped.geometry,
    )

    print(
        "Majority-green patch candidate cells:",
        int(patches["candidate"].sum()),
    )

    print(
        "Four-connected green patches:",
        patches["patch_count"],
    )

    print(
        "Accepted shared-edge connections:",
        patches["accepted_edges"],
    )

    print(
        "Patch mapped-green area distribution:",
        distribution(patches["patch_areas"]),
    )

    # ==================================================
    # 4. BUILD CANDIDATE INDICATOR DATASET
    # ==================================================

    print("\n[4] BUILD GREEN CONNECTIVITY DATASET")

    result = current[
        [
            "grid_id",
            "kmc_overlap_area_m2",
            "kmc_overlap_fraction",
            "valid_lulc_area_m2",
            "valid_lulc_coverage_fraction",
            "current_green_area_m2",
            "current_green_fraction",
            "mapped_green_share_lower_bound",
            "green_condition_status",
        ]
    ].copy()

    result["connectivity_source_year"] = 2025
    result["connectivity_source_product"] = "RAW_RF"

    for radius in RADII:

        context = contexts[radius]
        tag = f"{radius}m"

        result[
            f"green_neighbour_context_{tag}"
        ] = context["context"]

        result[
            f"green_neighbour_coverage_{tag}"
        ] = context["coverage"]

        result[
            f"green_neighbour_cell_count_{tag}"
        ] = context["counts"]

        result[
            f"green_neighbour_observed_count_{tag}"
        ] = context["observed_counts"]

        result[
            f"green_neighbour_valid_area_m2_{tag}"
        ] = context["valid_area"]

    result[
        "exploratory_majority_green_patch_candidate"
    ] = patches["candidate"]

    result[
        "green_patch_4connected_id"
    ] = patches["patch_ids"]

    result[
        "green_patch_4connected_cell_count"
    ] = patches["patch_cell_count"]

    result[
        "green_patch_4connected_mapped_area_km2"
    ] = patches["patch_green_area_km2"]

    # No final connectivity measure is selected here.

    # ==================================================
    # 5. VALIDATION
    # ==================================================

    print("\n[5] GREEN CONNECTIVITY QA")

    check(
        "9240_cells_preserved",
        len(result) == 9240
        and result.grid_id.is_unique,
    )

    for radius in RADII:

        context = contexts[radius]

        values = context["context"]
        coverage = context["coverage"]

        finite = np.isfinite(values)
        finite_coverage = np.isfinite(coverage)

        check(
            f"context_{radius}m_within_0_1",
            np.all(
                (values[finite] >= -1e-9)
                & (values[finite] <= 1 + 1e-9)
            ),
        )

        check(
            f"context_{radius}m_missing_only_without_valid_area",
            bool(
                np.array_equal(
                    finite,
                    context["valid_area"] > 1e-9,
                )
            ),
        )

        check(
            f"context_{radius}m_coverage_valid",
            np.all(
                (coverage[finite_coverage] >= -1e-9)
                & (coverage[finite_coverage] <= 1 + 1e-9)
            ),
        )

        check(
            f"context_{radius}m_reproduced",
            np.allclose(
                values[finite],
                (
                    context["green_area"][finite]
                    / context["valid_area"][finite]
                ),
                atol=1e-12,
            ),
        )

        check(
            f"context_{radius}m_no_focal_cell",
            bool(
                (context["counts"] < len(result)).all()
            ),
        )

    candidate = patches["candidate"]
    patch_ids = patches["patch_ids"]

    check(
        "patch_membership_matches_candidate",
        all(
            patch_ids[i] is not None
            if candidate[i]
            else patch_ids[i] is None
            for i in range(len(candidate))
        ),
    )

    check(
        "patch_area_positive",
        np.isfinite(
            patches["patch_green_area_km2"][candidate]
        ).all()
        and (
            patches["patch_green_area_km2"][candidate]
            > 0
        ).all(),
    )

    check(
        "patch_component_areas_reconcile",
        np.isclose(
            sum(patches["patch_areas"]),
            green_areas[candidate].sum() / 1e6,
            atol=1e-9,
        ),
    )

    check(
        "patch_count_matches_ids",
        len({
            x for x in patch_ids
            if x is not None
        }) == patches["patch_count"],
    )

    forbidden = [
        "pressure",
        "shap",
        "suitability",
        "ahp",
        "zoning",
        "priority",
    ]

    check(
        "no_forbidden_inputs",
        not any(
            term in col.lower()
            for col in result.columns
            for term in forbidden
        ),
    )

    if not all(checks.values()):
        raise SystemExit(
            "Stage 4.5 QA failed. No outputs exported."
        )

    # ==================================================
    # 6. EXPORT
    # ==================================================

    print("\n[6] EXPORT GREEN CONNECTIVITY")

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    result.to_csv(
        CSV_OUT,
        index=False,
    )

    geo = clipped[
        ["geometry"]
    ].reset_index().merge(
        result,
        on="grid_id",
        how="inner",
        validate="one_to_one",
    )

    if len(geo) != 9240:
        raise ValueError(
            "GIS merge lost grid cells"
        )

    geo.to_file(
        GPKG_OUT,
        layer="green_connectivity_2025_100m",
        driver="GPKG",
    )

    saved_csv = pd.read_csv(CSV_OUT)
    saved_gis = pyogrio.read_info(GPKG_OUT)

    check(
        "csv_export_9240",
        len(saved_csv) == 9240
        and saved_csv.grid_id.is_unique,
    )

    check(
        "gis_export_9240",
        saved_gis["features"] == 9240,
    )

    status = (
        "PASS_CONNECTIVITY_CANDIDATES"
        if all(checks.values())
        else "REVIEW_REQUIRED"
    )

    qa = {
        "section": "04_conservation_intelligence",
        "stage": "4.5",
        "status": status,
        "source_year": 2025,
        "source_product": "RAW_RF",
        "grid_cells": 9240,
        "neighbourhood_radii_m": RADII,
        "neighbourhood_definition": (
            "All other KMC grid cells whose full-grid "
            "centres fall within the selected radius. "
            "Observed green area divided by observed "
            "valid LULC area of those neighbouring cells."
        ),
        "neighbourhood_limitations": [
            "Grid-centre distance approximation, "
            "not exact circular-buffer land-area integration.",
            "Focal cell excluded to reduce duplication "
            "of current green condition.",
            "Unknown LULC coverage is not assigned "
            "zero green.",
            "Neighbourhood coverage retained for QA.",
        ],
        "neighbourhood_summary": {
            f"{radius}m": {
                "green_context_distribution": (
                    distribution(contexts[radius]["context"])
                ),
                "coverage_distribution": (
                    distribution(contexts[radius]["coverage"])
                ),
                "no_valid_context_cells": int(
                    np.isnan(
                        contexts[radius]["context"]
                    ).sum()
                ),
            }
            for radius in RADII
        },
        "patch_method": (
            "Exploratory four-neighbour 100m grid topology "
            "with positive shared clipped geometry edges"
        ),
        "patch_candidate_rule": (
            "Confirmed mapped green area divided by "
            "actual KMC-intersecting grid area > 0.50"
        ),
        "patch_rule_is_final_conservation_mask": False,
        "patch_candidate_cells": int(candidate.sum()),
        "patch_count": patches["patch_count"],
        "patch_area_distribution_km2": (
            distribution(patches["patch_areas"])
        ),
        "patch_limitations": [
            "A grid-based patch is not a mapped "
            "ecological habitat polygon.",
            "Green within adjacent cells may not "
            "physically touch at sub-grid scale.",
            "The 50 percent majority rule is exploratory.",
            "Other thresholds and connectivity rules "
            "require robustness analysis.",
            "No final Green Importance criterion "
            "has been chosen yet.",
        ],
        "checks": checks,
        "max_geometry_area_error_m2": (
            geometry_area_error
        ),
        "green_importance_calculated": False,
        "historical_threat_used": False,
        "future_pressure_used": False,
        "development_suitability_used": False,
    }

    QA_OUT.write_text(
        json.dumps(qa, indent=2),
        encoding="utf-8",
    )

    print("CSV:", CSV_OUT)
    print("GPKG:", GPKG_OUT)
    print("QA:", QA_OUT)

    print("\n" + "=" * 72)
    print("STAGE 4.5:", status)
    print("Grid cells:", len(result))
    print("Final connectivity indicator: NOT SELECTED")
    print("Green Importance: NOT CALCULATED")
    print("Future pressure used: NO")
    print("=" * 72)

    if status != "PASS_CONNECTIVITY_CANDIDATES":
        raise SystemExit(
            "Connectivity export validation failed."
        )


if __name__ == "__main__":
    main()
