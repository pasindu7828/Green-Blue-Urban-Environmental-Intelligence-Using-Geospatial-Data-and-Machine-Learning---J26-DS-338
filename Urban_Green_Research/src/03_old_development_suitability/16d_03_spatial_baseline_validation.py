
"""
STEP 16D-C - SPATIAL BASELINE VALIDATION AND INTERPRETATION

Descriptive internal validation of the provisional, equal-contribution
C1/C2/C4 baseline.

NOT independent ground truth, AHP, legal permission, or proof
of buildability.

All previous research inputs remain read-only.
"""

from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json

import geopandas as gpd
import numpy as np
import pandas as pd


# ============================================================
# 1. PROJECT PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

REPORTS = ROOT / "outputs/reports"

BASE_CSV = (
    REPORTS
    / "16d_b_provisional_suitability_baseline_100m.csv"
)

BASE_GPKG = (
    REPORTS
    / "16d_b_provisional_suitability_baseline_100m.gpkg"
)

BASE_SUMMARY = (
    REPORTS
    / "16d_b_provisional_baseline_summary.json"
)

BASE_CLASSES = (
    REPORTS
    / "16d_b_relative_suitability_class_summary.csv"
)

STANDARD_CSV = (
    REPORTS
    / "16d_a_provisional_standardized_criteria_100m.csv"
)

STANDARD_SUMMARY = (
    REPORTS
    / "16d_a_standardized_criteria_summary.json"
)

SCREEN_SUMMARY = (
    REPORTS
    / "16c_m_grid_zoning_screening_summary.json"
)

ZONING_SUMMARY = (
    REPORTS
    / "16c_j_zoning_reconciliation_summary.json"
)


# ============================================================
# 2. OUTPUT FILES
# ============================================================

OUT_PROFILE = (
    REPORTS
    / "16d_c_relative_class_criterion_profiles.csv"
)

OUT_REVIEW = (
    REPORTS
    / "16d_c_relative_class_planning_review.csv"
)

OUT_BLOCKS = (
    REPORTS
    / "16d_c_spatial_1km_block_patterns.csv"
)

OUT_CELLS = (
    REPORTS
    / "16d_c_grid_spatial_diagnostics_100m.csv"
)

OUT_SPATIAL = (
    REPORTS
    / "16d_c_spatial_autocorrelation.csv"
)

OUT_QA = (
    REPORTS
    / "16d_c_spatial_baseline_qa.csv"
)

OUT_SUMMARY = (
    REPORTS
    / "16d_c_spatial_baseline_summary.json"
)


# ============================================================
# 3. CONFIGURATION
# ============================================================

N_EXPECTED = 9240

TARGET_EPSG = 32644

PERMUTATIONS = 199

RANDOM_SEED = 42

# Numerical topology tolerance only.
# Not a legal planning buffer.
MIN_SHARED_EDGE_M = 0.001

RAW_FIELDS = {
    "c1_major_road_distance_m":
        "mean_road_distance_m",

    "c2_road_density_500m_km_per_km2":
        "mean_road_density_500m",

    "c3_context_building_pct_500m":
        "mean_building_context_pct",

    "c4_slope_mean_deg":
        "mean_slope_deg",

    "c5_elevation_mean_m":
        "mean_elevation_m",
}

CLASSES = [
    "RELATIVE_VERY_LOW",
    "RELATIVE_LOW",
    "RELATIVE_MODERATE",
    "RELATIVE_HIGH",
    "RELATIVE_VERY_HIGH",
]


# ============================================================
# 4. HELPER FUNCTIONS
# ============================================================

def sha256(path):

    h = hashlib.sha256()

    with path.open("rb") as source:

        for block in iter(
            lambda: source.read(1024 * 1024),
            b"",
        ):
            h.update(block)

    return h.hexdigest()


def read_json(path):

    with path.open(
        encoding="utf-8"
    ) as source:

        return json.load(source)


def read_csv_ids(path):

    table = pd.read_csv(
        path,
        dtype={"grid_id": "string"},
    )

    table["grid_id"] = (
        table["grid_id"].str.strip()
    )

    return table


def bool_series(series):

    strings = (
        series.astype("string")
        .str.strip()
        .str.lower()
    )

    accepted = strings.isin([
        "true", "false", "1", "0"
    ])

    if not bool(accepted.all()):

        raise ValueError(
            f"Unexpected boolean values in "
            f"{series.name}: "
            f"{strings[~accepted].unique().tolist()}"
        )

    return strings.isin([
        "true", "1"
    ]).to_numpy(dtype=bool)


# ============================================================
# 5. MORAN'S I FUNCTIONS
# ============================================================

def moran_i(values, left, right):

    x = np.asarray(
        values,
        dtype=float,
    )

    z = x - x.mean()

    denominator = float(
        np.dot(z, z)
    )

    edge_count = len(left)

    if edge_count == 0 or denominator <= 0:

        raise ValueError(
            "No usable spatial adjacency "
            "or no score variation."
        )

    # Undirected binary rook adjacency.
    # Each edge represents a shared polygon boundary.

    return float(
        (len(z) / edge_count)
        * (
            np.dot(
                z[left],
                z[right],
            )
            / denominator
        )
    )


def moran_permutations(values, left, right):

    observed = moran_i(
        values,
        left,
        right,
    )

    rng = np.random.default_rng(
        RANDOM_SEED
    )

    null_distribution = np.array([
        moran_i(
            rng.permutation(values),
            left,
            right,
        )
        for _ in range(PERMUTATIONS)
    ])

    # One-sided permutation evidence for
    # positive spatial clustering.

    p_positive = float(
        (
            1 + np.count_nonzero(
                null_distribution >= observed
            )
        )
        / (PERMUTATIONS + 1)
    )

    return (
        observed,
        float(null_distribution.mean()),
        p_positive,
    )


# ============================================================
# 6. MAIN
# ============================================================

def main():

    print("=" * 94)

    print(
        "STEP 16D-C - SPATIAL BASELINE "
        "VALIDATION AND INTERPRETATION"
    )

    print("=" * 94)

    sources = [
        BASE_CSV,
        BASE_GPKG,
        BASE_SUMMARY,
        BASE_CLASSES,
        STANDARD_CSV,
        STANDARD_SUMMARY,
        SCREEN_SUMMARY,
        ZONING_SUMMARY,
    ]

    for path in sources:

        if not path.is_file():

            raise FileNotFoundError(path)

    REPORTS.mkdir(
        parents=True,
        exist_ok=True,
    )

    hashes = {
        str(path): sha256(path)
        for path in sources
    }

    base_meta = read_json(
        BASE_SUMMARY
    )

    std_meta = read_json(
        STANDARD_SUMMARY
    )

    screen_meta = read_json(
        SCREEN_SUMMARY
    )

    zone_meta = read_json(
        ZONING_SUMMARY
    )

    checks = []

    def check(name, condition, detail=""):

        checks.append({
            "check": name,
            "passed": bool(condition),
            "detail": str(detail),
        })

    def checkpoint():

        failed = [
            record
            for record in checks
            if not record["passed"]
        ]

        if failed:

            pd.DataFrame(checks).to_csv(
                OUT_QA,
                index=False,
            )

            for record in failed:

                print(
                    "REVIEW:",
                    record["check"],
                    record["detail"],
                )

            raise RuntimeError(
                "Step 16D-C validation needs review. "
                "See QA CSV."
            )

    # ========================================================
    # 7. VERIFY COMPLETED RESEARCH STEPS
    # ============================================================

    check(
        "16D-B, 16D-A, 16C-M and 16C-J passed",

        all(
            item.get("qa_status") == "PASS"
            for item in [
                base_meta,
                std_meta,
                screen_meta,
                zone_meta,
            ]
        ),
    )

    check(
        "Reference baseline remains explicitly provisional",

        base_meta.get("methodology_status")
        == "PROVISIONAL_REFERENCE_BASELINE_ONLY"

        and base_meta.get(
            "expert_ahp_weights_calculated"
        ) is False

        and base_meta.get(
            "final_suitability_model_calculated"
        ) is False,
    )

    check(
        "Inputs describe the same 9240-cell study area",

        base_meta.get("grid_count")
        == std_meta.get("grid_cells")
        == screen_meta.get("grid_count")
        == N_EXPECTED,
    )

    checkpoint()

    # ========================================================
    # 8. LOAD GRID DATA
    # ============================================================

    print("\n=== LOADING EXISTING RESEARCH DATA ===")

    base = read_csv_ids(
        BASE_CSV
    )

    standard = read_csv_ids(
        STANDARD_CSV
    )

    original_classes = pd.read_csv(
        BASE_CLASSES
    )

    geodata = gpd.read_file(
        BASE_GPKG,
        layer="provisional_suitability_baseline",
    )

    geodata["grid_id"] = (
        geodata["grid_id"]
        .astype("string")
        .str.strip()
    )

    check(
        "CSV, standardized criteria and GIS have 9240 unique IDs",

        all(
            len(df) == N_EXPECTED
            and df["grid_id"].notna().all()
            and df["grid_id"].is_unique

            for df in [
                base,
                standard,
                geodata,
            ]
        ),
    )

    check(
        "All grid ID sets match",

        set(base["grid_id"])
        == set(standard["grid_id"])
        == set(geodata["grid_id"]),
    )

    check(
        "Baseline GIS has valid EPSG:32644 polygons",

        geodata.crs is not None

        and geodata.crs.to_epsg()
        == TARGET_EPSG

        and geodata.geometry.notna().all()

        and (
            ~geodata.geometry.is_empty
        ).all()

        and geodata.geometry.is_valid.all(),
    )

    checkpoint()

    # ========================================================
    # 9. ALIGN DATA BY GRID ID
    # ============================================================

    std = (
        standard
        .set_index("grid_id")
        .reindex(base["grid_id"])
    )

    polygons = (
        geodata
        .set_index("grid_id")
        .geometry
        .reindex(base["grid_id"])
    )

    check(
        "All aligned geometries exist",
        polygons.notna().all(),
    )

    check(
        "Raw C1-C5 columns exist",

        set(RAW_FIELDS).issubset(
            std.columns
        ),
    )

    required_base = [
        "baseline_pc",
        "baseline_rank",
        "relative_class",
        "c1_pc",
        "c2_pc",
        "c4_pc",
        "cell_area_m2",
        "review_needed",
        "conservation_water_pct",
        "ambiguous_pct",
        "highland_approx_pct",
        "classified_pct",
        "unclassified_pct",
        "unmapped_pct",
        "small_boundary_fragment",
        "top_decile_pc",
        "top_decile_rank",
        "zoning_legal_status",
        "legal_permission",
    ]

    check(
        "Baseline has all required audit fields",

        set(required_base).issubset(
            base.columns
        ),
    )

    checkpoint()

    # ========================================================
    # 10. CONVERT NUMERIC VALUES
    # ============================================================

    score_cols = [
        "baseline_pc",
        "baseline_rank",
        "c1_pc",
        "c2_pc",
        "c4_pc",
    ]

    percentage_cols = [
        "conservation_water_pct",
        "ambiguous_pct",
        "highland_approx_pct",
        "classified_pct",
        "unclassified_pct",
        "unmapped_pct",
    ]

    for column in (
        score_cols
        + ["cell_area_m2"]
        + percentage_cols
    ):

        base[column] = pd.to_numeric(
            base[column],
            errors="coerce",
        )

    for field in RAW_FIELDS:

        base[field] = pd.to_numeric(
            std[field].to_numpy(),
            errors="coerce",
        )

    numeric_cols = (
        score_cols
        + list(RAW_FIELDS)
        + ["cell_area_m2"]
        + percentage_cols
    )

    check(
        "All spatial diagnostic input values finite",

        np.isfinite(
            base[numeric_cols]
            .to_numpy(dtype=float)
        ).all(),
    )

    check(
        "Baseline and criterion scores between zero and one",

        base[score_cols].ge(0).all().all()

        and base[score_cols].le(1).all().all(),
    )

    check(
        "Baseline equals C1/C2/C4 diagnostic mean",

        np.allclose(
            base["baseline_pc"],

            base[
                [
                    "c1_pc",
                    "c2_pc",
                    "c4_pc",
                ]
            ].mean(axis=1),

            atol=1e-10,
        ),
    )

    check(
        "Relative classes are the expected five levels",

        set(
            base["relative_class"]
        ) == set(CLASSES),
    )

    check(
        "Zoning statuses unverified and permissions unassessed",

        base["zoning_legal_status"]
        .astype(str)
        .eq("UNVERIFIED")
        .all()

        and base["legal_permission"]
        .astype(str)
        .eq("NOT_ASSESSED")
        .all(),
    )

    checkpoint()

    # ========================================================
    # 11. GEOMETRY AND ZONING VALIDATION
    # ============================================================

    print("\n=== VALIDATING GRID GEOMETRY ===")

    area = (
        base["cell_area_m2"]
        .to_numpy(dtype=float)
    )

    geom = gpd.GeoSeries(
        polygons.to_numpy(),
        crs=geodata.crs,
    )

    geom_area = (
        geom.area.to_numpy(dtype=float)
    )

    review = bool_series(
        base["review_needed"]
    )

    small = bool_series(
        base["small_boundary_fragment"]
    )

    high_pc = bool_series(
        base["top_decile_pc"]
    )

    high_rank = bool_series(
        base["top_decile_rank"]
    )

    max_area_error = float(
        np.max(
            np.abs(
                area - geom_area
            )
        )
    )

    check(
        "Polygon areas match recorded clipped-cell areas",

        max_area_error <= 0.01,

        f"Max difference: {max_area_error:.8f} m2",
    )

    check(
        "Grid area matches reconciled municipal area",

        abs(
            area.sum() / 10000
            - float(zone_meta["kmc_area_ha"])
        ) <= 0.05,
    )

    check(
        "Boundary fragment flags agree with actual areas",

        np.array_equal(
            small,
            area < 5000.0,
        ),
    )

    check(
        "Top-decile fields agree with existing baseline",

        np.array_equal(
            high_pc,

            base["baseline_pc"].to_numpy()
            >= base["baseline_pc"].quantile(0.90)
            - 1e-12,
        )

        and np.array_equal(
            high_rank,

            base["baseline_rank"].to_numpy()
            >= base["baseline_rank"].quantile(0.90)
            - 1e-12,
        ),
    )

    partition = (
        base[
            [
                "classified_pct",
                "unclassified_pct",
                "unmapped_pct",
            ]
        ].sum(axis=1)
    )

    check(
        "Exclusive zoning percentages total 100",

        np.max(
            np.abs(
                partition - 100
            )
        ) <= 0.001,

        "Tolerance allows CSV rounding only",
    )

    # Verify zoning coverage by integrating
    # percentages over actual grid-cell areas.

    zoning_checks = [
        (
            "classified_pct",
            "classified_area_ha",
        ),
        (
            "unclassified_pct",
            "unclassified_only_area_ha",
        ),
        (
            "unmapped_pct",
            "truly_unmapped_area_ha",
        ),
    ]

    for pct_col, meta_key in zoning_checks:

        actual = float(
            np.sum(
                area
                * base[pct_col].to_numpy()
                / 100
            )
            / 10000
        )

        previous = float(
            zone_meta[meta_key]
        )

        check(
            f"Area-weighted {pct_col} reproduces 16C-J",

            abs(actual - previous) <= 0.05,

            (
                f"Grid: {actual:.4f} ha; "
                f"previous: {previous:.4f} ha"
            ),
        )

    checkpoint()

    # ========================================================
    # 12. EXTRACT ORIGINAL ROW/COLUMN COORDINATES
    # ============================================================

    print("\n=== BUILDING SPATIAL NEIGHBOUR STRUCTURE ===")

    rc = base["grid_id"].str.extract(
        r"^G100_R(\d+)_C(\d+)$"
    )

    check(
        "Every grid ID has valid 100m row-column format",

        rc.notna().all().all(),
    )

    checkpoint()

    rows = rc[0].astype(int).to_numpy()

    cols = rc[1].astype(int).to_numpy()

    coords = list(
        zip(
            rows.tolist(),
            cols.tolist(),
        )
    )

    check(
        "Grid row-column pairs are unique",

        len(set(coords)) == N_EXPECTED,
    )

    checkpoint()

    coordinate_lookup = {
        cell: i
        for i, cell in enumerate(coords)
    }

    left = []
    right = []

    # Candidate rook neighbours:
    # one row below or one column to the right.
    # Each undirected pair is visited once.

    for i, (r, c) in enumerate(coords):

        for neighbour in [
            (r + 1, c),
            (r, c + 1),
        ]:

            other = coordinate_lookup.get(
                neighbour
            )

            if other is not None:

                left.append(i)
                right.append(other)

    left = np.asarray(
        left,
        dtype=int,
    )

    right = np.asarray(
        right,
        dtype=int,
    )

    # Check actual shared polygon boundaries.
    # This handles clipped boundary cells correctly.
    # A corner-only contact is not rook adjacency.

    edge_lengths = (
        geom.iloc[left]
        .reset_index(drop=True)
        .boundary
        .intersection(
            geom.iloc[right]
            .reset_index(drop=True)
            .boundary,
            align=False,
        )
        .length
        .to_numpy(dtype=float)
    )

    actual_rook = (
        edge_lengths > MIN_SHARED_EDGE_M
    )

    left = left[actual_rook]
    right = right[actual_rook]

    neighbours = np.bincount(
        np.r_[left, right],
        minlength=N_EXPECTED,
    )

    check(
        "Real shared-edge grid neighbours exist",

        len(left) > N_EXPECTED,

        f"Edges: {len(left)}",
    )

    check(
        "Adjacency links connect different cells",

        np.all(left != right),
    )

    checkpoint()

    # ========================================================
    # 13. SPATIAL AUTOCORRELATION
    # ============================================================

    print("\n=== SPATIAL AUTOCORRELATION ===")

    spatial_rows = []

    for method in [
        "baseline_pc",
        "baseline_rank",
    ]:

        observed, mean_perm, p_positive = (
            moran_permutations(
                base[method].to_numpy(
                    dtype=float
                ),
                left,
                right,
            )
        )

        spatial_rows.append({
            "scenario": method,

            "grid_cells": N_EXPECTED,

            "undirected_rook_edges":
                len(left),

            "isolated_cells": int(
                (neighbours == 0).sum()
            ),

            "observed_moran_i":
                observed,

            "permutation_mean_moran_i":
                mean_perm,

            "positive_clustering_permutation_p":
                p_positive,

            "permutations":
                PERMUTATIONS,

            "random_seed":
                RANDOM_SEED,

            "meaning":
                "INTERNAL_SPATIAL_STRUCTURE_ONLY_NOT_EXTERNAL_VALIDATION",
        })

    spatial = pd.DataFrame(
        spatial_rows
    )

    check(
        "Spatial autocorrelation values finite",

        np.isfinite(
            spatial[
                [
                    "observed_moran_i",
                    "permutation_mean_moran_i",
                    "positive_clustering_permutation_p",
                ]
            ].to_numpy(dtype=float)
        ).all(),
    )

    # ========================================================
    # 14. HIGH-CLASS SPATIAL CONNECTIONS
    # ============================================================

    same_class = (
        base["relative_class"].to_numpy()
        == "RELATIVE_VERY_HIGH"
    )

    selected = (
        same_class[left]
        & same_class[right]
    )

    high_high_edges = int(
        selected.sum()
    )

    high_high_neighbors = np.zeros(
        N_EXPECTED,
        dtype=int,
    )

    np.add.at(
        high_high_neighbors,
        left[selected],
        1,
    )

    np.add.at(
        high_high_neighbors,
        right[selected],
        1,
    )

    # ========================================================
    # 15. SAVE PER-CELL SPATIAL DIAGNOSTICS
    # ============================================================

    cell_data = pd.DataFrame({
        "grid_id":
            base["grid_id"],

        "grid_row":
            rows,

        "grid_col":
            cols,

        "nominal_1km_block": [
            f"R{r // 10:03d}_C{c // 10:03d}"
            for r, c in coords
        ],

        "baseline_pc":
            base["baseline_pc"],

        "relative_class":
            base["relative_class"],

        "rook_neighbor_count":
            neighbours,

        "very_high_rook_neighbor_count":
            high_high_neighbors,

        "top_decile_both_methods":
            high_pc & high_rank,

        "planning_review":
            review,

        "small_boundary_fragment":
            small,

        "cell_area_m2":
            area,
    })

    # ========================================================
    # 16. CRITERION AND PLANNING-REVIEW PROFILES
    # ============================================================

    print("\n=== CALCULATING CRITERION PROFILES ===")

    work = base.copy()

    work["review_boolean"] = review
    work["small_boolean"] = small

    work["conservation_boolean"] = (
        work["conservation_water_pct"] > 0
    )

    work["ambiguous_boolean"] = (
        work["ambiguous_pct"] > 0
    )

    work["highland_boolean"] = (
        work["highland_approx_pct"] > 0
    )

    work["stable_top_decile"] = (
        high_pc & high_rank
    )

    records_profile = []
    records_review = []

    for label in CLASSES:

        x = work.loc[
            work["relative_class"] == label
        ]

        if x.empty:
            continue

        record = {
            "relative_class":
                label,

            "grid_cells":
                len(x),

            "area_ha":
                float(
                    x["cell_area_m2"].sum()
                    / 10000
                ),

            "mean_baseline":
                float(
                    x["baseline_pc"].mean()
                ),

            "median_baseline":
                float(
                    x["baseline_pc"].median()
                ),

            "mean_c1_standardized":
                float(
                    x["c1_pc"].mean()
                ),

            "mean_c2_standardized":
                float(
                    x["c2_pc"].mean()
                ),

            "mean_c4_standardized":
                float(
                    x["c4_pc"].mean()
                ),
        }

        for field, output_name in RAW_FIELDS.items():

            record[output_name] = float(
                x[field].mean()
            )

        records_profile.append(
            record
        )

        records_review.append({
            "relative_class":
                label,

            "grid_cells":
                len(x),

            "review_cells":
                int(
                    x["review_boolean"].sum()
                ),

            "review_cell_pct":
                float(
                    x["review_boolean"].mean()
                    * 100
                ),

            "conservation_water_review_cells":
                int(
                    x["conservation_boolean"].sum()
                ),

            "ambiguous_name_review_cells":
                int(
                    x["ambiguous_boolean"].sum()
                ),

            "approximate_highland_review_cells":
                int(
                    x["highland_boolean"].sum()
                ),

            "small_boundary_fragments":
                int(
                    x["small_boolean"].sum()
                ),

            "stable_top_decile_both_methods":
                int(
                    x["stable_top_decile"].sum()
                ),

            "screening_only_not_legal_exclusion":
                True,
        })

    profiles = pd.DataFrame(
        records_profile
    )

    reviews = pd.DataFrame(
        records_review
    )

    # ========================================================
    # 17. NOMINAL 1-KM SPATIAL BLOCK PATTERNS
    # ============================================================

    # A 10x10 group of nominal 100m cells.
    # These are NOT GN divisions or legal planning zones.

    blocks = (
        cell_data
        .groupby(
            "nominal_1km_block",
            as_index=False,
        )
        .agg(
            grid_cells=(
                "grid_id",
                "size",
            ),

            covered_area_ha=(
                "cell_area_m2",
                lambda x: float(
                    x.sum() / 10000
                ),
            ),

            mean_baseline=(
                "baseline_pc",
                "mean",
            ),

            very_high_cell_count=(
                "relative_class",
                lambda x: int(
                    (
                        x == "RELATIVE_VERY_HIGH"
                    ).sum()
                ),
            ),

            planning_review_cells=(
                "planning_review",
                "sum",
            ),

            small_fragments=(
                "small_boundary_fragment",
                "sum",
            ),

            stable_top_decile_cells=(
                "top_decile_both_methods",
                "sum",
            ),
        )
    )

    blocks["review_cell_pct"] = (
        100
        * blocks["planning_review_cells"]
        / blocks["grid_cells"]
    )

    blocks = (
        blocks
        .sort_values(
            [
                "mean_baseline",
                "grid_cells",
            ],
            ascending=[
                False,
                False,
            ],
        )
        .reset_index(drop=True)
    )

    # ========================================================
    # 18. FINAL QA
    # ============================================================

    check(
        "Five class profiles cover all grid cells",

        len(profiles) == 5

        and profiles["grid_cells"].sum()
        == N_EXPECTED,
    )

    check(
        "Class profile areas reproduce the KMC area",

        abs(
            profiles["area_ha"].sum()
            - float(zone_meta["kmc_area_ha"])
        ) <= 0.05,
    )

    check(
        "Planning review count matches Step 16D-B",

        int(review.sum())
        == base_meta.get("planning_review_cells"),
    )

    check(
        "Small fragment count matches Step 16D-B",

        int(small.sum())
        == base_meta.get("small_boundary_fragments"),
    )

    check(
        "Spatial blocks cover each grid exactly once",

        blocks["grid_cells"].sum()
        == N_EXPECTED,
    )

    check(
        "Geometry diagnostic covers every grid ID",

        len(cell_data) == N_EXPECTED

        and cell_data["grid_id"].is_unique,
    )

    # Compare class statistics with frozen Step 16D-B.

    observed = (
        profiles
        .set_index("relative_class")
        .sort_index()
    )

    reference = (
        original_classes
        .set_index("relative_class")
        .sort_index()
    )

    check(
        "Class counts reproduce Step 16D-B report",

        observed.index.equals(
            reference.index
        )

        and np.array_equal(
            observed["grid_cells"].to_numpy(),
            reference["grid_cells"].to_numpy(),
        ),
    )

    check(
        "Class areas reproduce Step 16D-B report",

        observed.index.equals(
            reference.index
        )

        and np.allclose(
            observed["area_ha"],
            reference["area_ha"],
            atol=0.0002,
        ),
    )

    check(
        "Input files unchanged",

        all(
            sha256(path)
            == hashes[str(path)]
            for path in sources
        ),
    )

    checkpoint()

    # ========================================================
    # 19. SAVE REPORTS
    # ============================================================

    output_frames = [
        (profiles, OUT_PROFILE),
        (reviews, OUT_REVIEW),
        (blocks, OUT_BLOCKS),
        (cell_data, OUT_CELLS),
        (spatial, OUT_SPATIAL),
        (pd.DataFrame(checks), OUT_QA),
    ]

    for frame, path in output_frames:

        frame.to_csv(
            path,
            index=False,
        )

    # ========================================================
    # 20. SUMMARY JSON
    # ============================================================

    summary = {
        "step": "16D-C",

        "qa_status": "PASS",

        "methodology_status":
            "PROVISIONAL_BASELINE_INTERNAL_DIAGNOSTICS_ONLY",

        "generated_utc": datetime.now(
            timezone.utc
        ).isoformat(),

        "grid_count": N_EXPECTED,

        "study_area_ha": float(
            area.sum() / 10000
        ),

        "spatial_rook_edges":
            len(left),

        "isolated_grid_cells": int(
            (neighbours == 0).sum()
        ),

        "very_high_to_very_high_rook_edges":
            high_high_edges,

        "positive_clustering_permutations":
            PERMUTATIONS,

        "moran_pc": float(
            spatial.loc[
                spatial["scenario"] == "baseline_pc",
                "observed_moran_i",
            ].iloc[0]
        ),

        "moran_rank": float(
            spatial.loc[
                spatial["scenario"] == "baseline_rank",
                "observed_moran_i",
            ].iloc[0]
        ),

        "high_class_review_cells": int(
            reviews.loc[
                reviews["relative_class"]
                == "RELATIVE_VERY_HIGH",
                "review_cells",
            ].iloc[0]
        ),

        "high_class_total_cells": int(
            reviews.loc[
                reviews["relative_class"]
                == "RELATIVE_VERY_HIGH",
                "grid_cells",
            ].iloc[0]
        ),

        "planning_review_cells": int(
            review.sum()
        ),

        "small_boundary_fragments": int(
            small.sum()
        ),

        "blocks_1km_nominal":
            len(blocks),

        "qa_checks":
            len(checks),

        "qa_passed": int(
            pd.DataFrame(checks)["passed"].sum()
        ),

        "input_sha256": {
            path.name: hashes[str(path)]
            for path in sources
        },

        "limitations": [
            (
                "Internal spatial autocorrelation "
                "is not ground-truth suitability validation."
            ),

            (
                "Relative classes are within-Kaduwela "
                "rankings, not legal suitability levels."
            ),

            (
                "Provisional zoning flags are review "
                "indicators, not automatic exclusions."
            ),

            (
                "Nominal 1-km blocks are spatial "
                "summaries, not GN divisions or parcels."
            ),

            (
                "Small boundary fragments cannot be "
                "interpreted as independently buildable."
            ),

            (
                "C3 and C5 remain outside the baseline; "
                "expert AHP judgments are not supplied."
            ),

            (
                "All metrics describe an equal-contribution "
                "provisional baseline."
            ),
        ],

        "outputs": [
            path.name

            for path in [
                OUT_PROFILE,
                OUT_REVIEW,
                OUT_BLOCKS,
                OUT_CELLS,
                OUT_SPATIAL,
                OUT_QA,
                OUT_SUMMARY,
            ]
        ],
    }

    with OUT_SUMMARY.open(
        "w",
        encoding="utf-8",
    ) as destination:

        json.dump(
            summary,
            destination,
            ensure_ascii=False,
            indent=2,
        )

    # ========================================================
    # 21. TERMINAL REPORT
    # ============================================================

    print("\n=== RELATIVE CLASS PROFILES ===")

    print(
        profiles[
            [
                "relative_class",
                "grid_cells",
                "mean_baseline",
                "mean_road_distance_m",
                "mean_road_density_500m",
                "mean_slope_deg",
            ]
        ]
        .round(3)
        .to_string(index=False)
    )

    print("\n=== CLASS PLANNING REVIEW ===")

    print(
        reviews[
            [
                "relative_class",
                "review_cells",
                "review_cell_pct",
                "conservation_water_review_cells",
                "small_boundary_fragments",
            ]
        ]
        .round(2)
        .to_string(index=False)
    )

    print(
        "\n=== SPATIAL AUTOCORRELATION (INTERNAL) ==="
    )

    print(
        spatial[
            [
                "scenario",
                "undirected_rook_edges",
                "isolated_cells",
                "observed_moran_i",
                "positive_clustering_permutation_p",
            ]
        ]
        .round(4)
        .to_string(index=False)
    )

    print(
        "\n=== FIVE HIGHER-MEAN POPULATED "
        "1-KM NOMINAL BLOCKS (>=20 CELLS) ==="
    )

    print(
        blocks.loc[
            blocks["grid_cells"] >= 20,

            [
                "nominal_1km_block",
                "grid_cells",
                "mean_baseline",
                "review_cell_pct",
            ],
        ]
        .head(5)
        .round(4)
        .to_string(index=False)
    )

    print("\n=== QA RESULTS ===")

    for record in checks:

        prefix = (
            "PASS: "
            if record["passed"]
            else "REVIEW: "
        )

        print(
            prefix + record["check"]
        )

    print(
        f"\nOverall QA: PASS "
        f"({len(checks)}/{len(checks)})"
    )

    print("\n=== GENERATED REPORTS ===")

    for path in [
        OUT_PROFILE,
        OUT_REVIEW,
        OUT_BLOCKS,
        OUT_CELLS,
        OUT_SPATIAL,
        OUT_QA,
        OUT_SUMMARY,
    ]:

        print(
            path.relative_to(ROOT)
        )

    print(
        "\nNo expert AHP, final suitability map, "
        "or legal approval calculated."
    )

    print("=" * 94)


# ============================================================
# 22. EXECUTE
# ============================================================

if __name__ == "__main__":

    main()
