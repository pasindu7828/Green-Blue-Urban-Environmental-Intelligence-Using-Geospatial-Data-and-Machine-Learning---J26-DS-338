
"""
STEP 16C-M (CORRECTED) - GRID-LEVEL PROVISIONAL ZONING SCREENING

Use the original Step 16B boundary-clipped 100 m suitability grid.
Validate it against the modelling grid AFTER clipping modelling cells
by the supplied Kaduwela Municipal Council boundary.

Produce provisional zoning and conservation REVIEW indicators, not
legal permission labels or weighted suitability scores.

Read-only original sources; all outputs go to outputs/reports/.
"""

from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import os
import sqlite3

import geopandas as gpd
import numpy as np
import pandas as pd


# ============================================================
# 1. PROJECT PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

REPORTS = ROOT / "outputs/reports"

RAW_CSV = (
    ROOT
    / "data/processed/development_suitability"
    / "raw_criteria/16b_raw_suitability_criteria_100m.csv"
)

SUITABILITY_GPKG = (
    ROOT
    / "data/processed/development_suitability"
    / "raw_criteria/16b_raw_suitability_criteria_100m.gpkg"
)

MODELLING_GPKG = (
    ROOT
    / "data/processed/urbanization_pressure"
    / "grid/kaduwela_100m_modelling_grid.gpkg"
)

UDA_DIR = (
    ROOT
    / "data/raw/uda/zoning_provisional"
    / "provided_package_extracted/New data"
)

ZONE_GPKG = (
    UDA_DIR / "Kaduwela_UDA_2024_AUDITED_PROVISIONAL.gpkg"
)

SCREEN_GPKG = (
    UDA_DIR
    / "Kaduwela_Conservation_Review_Screening_PROVISIONAL.gpkg"
)

BOUNDARY_GPKG = (
    UDA_DIR / "Kaduwela_MC_Boundary.gpkg"
)

HIGHLAND_GEOJSON = (
    UDA_DIR
    / "Kaduwela_2_High_Land_Conservation_Zones_APPROXIMATE.geojson"
)

SUMMARY_J = (
    REPORTS / "16c_j_zoning_reconciliation_summary.json"
)

SUMMARY_L = (
    REPORTS / "16c_l_expert_review_summary.json"
)

SUMMARY_M1 = (
    REPORTS / "16c_m1_grid_geometry_summary.json"
)

SOURCE_INVENTORY = (
    REPORTS / "16c_h_source_inventory.csv"
)


# ============================================================
# 2. OUTPUT PATHS
# ============================================================

OUT_CSV = (
    REPORTS / "16c_m_grid_zoning_screening_100m.csv"
)

OUT_GPKG = (
    REPORTS / "16c_m_grid_zoning_screening_100m.gpkg"
)

OUT_GROUPS = (
    REPORTS / "16c_m_grid_screening_categories.csv"
)

OUT_QA = (
    REPORTS / "16c_m_grid_zoning_screening_qa.csv"
)

OUT_SUMMARY = (
    REPORTS / "16c_m_grid_zoning_screening_summary.json"
)


# ============================================================
# 3. CONFIGURATION
# ============================================================

EXPECTED_CELLS = 9240

EPSG = 32644

# Geometry measurement tolerances only.
# These are NOT planning or legal thresholds.

GEOMETRY_MATCH_TOLERANCE_M2 = 1.0

MIN_REVIEW_INTERSECTION_M2 = 0.01

MIN_ZONE_INTERSECTION_M2 = 1e-8

PURE_CELL_FRACTION = 0.999

AREA_RECONCILIATION_HA = 0.05


# ============================================================
# 4. HELPER FUNCTIONS
# ============================================================

def sha256(path):

    digest = hashlib.sha256()

    with path.open("rb") as file:

        for chunk in iter(
            lambda: file.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def read_json(path):

    with path.open(
        encoding="utf-8"
    ) as file:

        return json.load(file)


def union_all(geometry_series):

    if hasattr(geometry_series, "union_all"):

        return geometry_series.union_all()

    return geometry_series.unary_union


def single_grid_layer(path):

    with sqlite3.connect(str(path)) as conn:

        records = conn.execute(
            "SELECT table_name "
            "FROM gpkg_contents "
            "WHERE data_type = 'features' "
            "ORDER BY table_name"
        ).fetchall()

    names = [
        record[0]
        for record in records
    ]

    if len(names) == 1:
        return names[0]

    candidates = [
        name
        for name in names
        if "grid" in name.lower()
        or "100m" in name.lower()
    ]

    if len(candidates) != 1:

        raise RuntimeError(
            "Cannot identify one grid layer in "
            f"{path.name}: {names}"
        )

    return candidates[0]


def read_grid(path):

    layer = single_grid_layer(path)

    gdf = gpd.read_file(
        path,
        layer=layer,
    )

    id_columns = [
        column
        for column in gdf.columns
        if column.lower() == "grid_id"
    ]

    if len(id_columns) != 1:

        raise RuntimeError(
            f"grid_id not uniquely identified "
            f"in {path.name}: {list(gdf.columns)}"
        )

    gdf = gdf.rename(
        columns={
            id_columns[0]: "grid_id"
        }
    )

    gdf["grid_id"] = (
        gdf["grid_id"]
        .astype("string")
        .str.strip()
    )

    return gdf, layer


def validated_projected(gdf, label):

    if gdf.crs is None:

        raise RuntimeError(
            f"{label}: missing CRS"
        )

    geometry_valid = bool(
        gdf.geometry.notna().all()
        and (~gdf.geometry.is_empty).all()
        and gdf.geometry.is_valid.all()
    )

    if not geometry_valid:

        raise RuntimeError(
            f"{label}: invalid, empty or null geometry"
        )

    return (
        gdf
        .to_crs(epsg=EPSG)
        .reset_index(drop=True)
    )


def geometry_overlap_area(geometries, polygon):

    return (
        geometries
        .intersection(polygon)
        .area
        .to_numpy(dtype=float)
    )


# ============================================================
# 5. MAIN
# ============================================================

def main():

    print("=" * 95)

    print(
        "STEP 16C-M - CORRECTED GRID-LEVEL "
        "PROVISIONAL ZONING SCREENING"
    )

    print("=" * 95)

    inputs = [
        RAW_CSV,
        SUITABILITY_GPKG,
        MODELLING_GPKG,
        ZONE_GPKG,
        SCREEN_GPKG,
        BOUNDARY_GPKG,
        HIGHLAND_GEOJSON,
        SUMMARY_J,
        SUMMARY_L,
        SUMMARY_M1,
        SOURCE_INVENTORY,
    ]

    for path in inputs:

        if not path.is_file():

            raise FileNotFoundError(
                path
            )

    REPORTS.mkdir(
        parents=True,
        exist_ok=True,
    )

    previous_j = read_json(
        SUMMARY_J
    )

    previous_l = read_json(
        SUMMARY_L
    )

    previous_m1 = read_json(
        SUMMARY_M1
    )

    source_digests = {
        str(path): sha256(path)
        for path in inputs
    }

    checks = []

    def check(label, passed, detail=""):

        checks.append({
            "check": label,
            "passed": bool(passed),
            "detail": str(detail),
        })

    def stop_on_fail():

        failed = [
            row
            for row in checks
            if not row["passed"]
        ]

        if failed:

            pd.DataFrame(checks).to_csv(
                OUT_QA,
                index=False,
            )

            for row in failed:

                print(
                    "REVIEW:",
                    row["check"],
                    "-",
                    row["detail"],
                )

            raise RuntimeError(
                "Step 16C-M QA requires review. "
                "See QA CSV. Sources unchanged."
            )

    # ========================================================
    # 6. VERIFY PREVIOUS STEPS
    # ========================================================

    check(
        "Step 16C-J passed",
        previous_j.get("qa_status") == "PASS",
    )

    check(
        "Step 16C-L passed",
        previous_l.get("qa_status") == "PASS",
    )

    check(
        "Step 16C-M1 diagnostic passed",
        previous_m1.get("qa_status") == "PASS",
    )

    check(
        "Frozen Step 16B CSV matches earlier SHA256",

        source_digests[str(RAW_CSV)]
        == previous_j.get("frozen_input_sha256")
        == previous_l.get("frozen_input_sha256"),
    )

    check(
        "Expert questionnaire has not assigned weights",
        previous_l.get("weights_assigned") is False,
    )

    check(
        "Previous diagnostic confirmed 9240 cells",
        previous_m1.get("grid_count")
        == EXPECTED_CELLS,
    )

    check(
        "Suitability GeoPackage unchanged since M1",

        source_digests[str(SUITABILITY_GPKG)]
        == previous_m1.get(
            "source_hashes", {}
        ).get(SUITABILITY_GPKG.name),
    )

    check(
        "Modelling GeoPackage unchanged since M1",

        source_digests[str(MODELLING_GPKG)]
        == previous_m1.get(
            "source_hashes", {}
        ).get(MODELLING_GPKG.name),
    )

    # ========================================================
    # 7. VERIFY PROVISIONAL GIS SOURCES
    # ========================================================

    inventory = pd.read_csv(
        SOURCE_INVENTORY,
        dtype="string",
    )

    for path in [
        ZONE_GPKG,
        SCREEN_GPKG,
        BOUNDARY_GPKG,
        HIGHLAND_GEOJSON,
    ]:

        record = inventory[
            inventory["file"] == path.name
        ]

        matched = (
            len(record) == 1
            and str(record.iloc[0]["sha256"])
            == source_digests[str(path)]
        )

        check(
            "UDA source unchanged: " + path.name,
            matched,
        )

    stop_on_fail()

    # ========================================================
    # 8. LOAD EXISTING RESEARCH GRIDS
    # ========================================================

    print(
        "\n=== VALIDATING ORIGINAL GRID GEOMETRY ==="
    )

    grid, grid_layer = read_grid(
        SUITABILITY_GPKG
    )

    full_grid, modelling_layer = read_grid(
        MODELLING_GPKG
    )

    raw_ids = pd.read_csv(
        RAW_CSV,
        usecols=["grid_id"],
        dtype={"grid_id": "string"},
    )["grid_id"].str.strip()

    check(
        "Suitability has 9240 unique IDs",

        len(grid) == EXPECTED_CELLS
        and grid["grid_id"].notna().all()
        and grid["grid_id"].is_unique,
    )

    check(
        "Modelling has 9240 unique IDs",

        len(full_grid) == EXPECTED_CELLS
        and full_grid["grid_id"].notna().all()
        and full_grid["grid_id"].is_unique,
    )

    check(
        "Frozen CSV has 9240 unique IDs",

        len(raw_ids) == EXPECTED_CELLS
        and raw_ids.notna().all()
        and raw_ids.is_unique,
    )

    check(
        "CSV, suitability and modelling grid IDs match",

        set(grid["grid_id"])
        == set(full_grid["grid_id"])
        == set(raw_ids),
    )

    stop_on_fail()

    grid = validated_projected(
        grid,
        "Step 16B suitability grid",
    )

    full_grid = validated_projected(
        full_grid,
        "Full modelling grid",
    )

    boundary_gdf = validated_projected(
        gpd.read_file(
            BOUNDARY_GPKG,
            layer="Kaduwela_MC_Boundary",
        ),
        "KMC boundary",
    )

    kmc_boundary = union_all(
        boundary_gdf.geometry
    )

    # ========================================================
    # 9. CORRECTED GRID GEOMETRY COMPARISON
    # ========================================================

    # The original modelling grid contains full squares.
    # Step 16B stores boundary-clipped polygons.
    #
    # Therefore compare:
    #
    # Step 16B clipped grid
    #
    # against:
    #
    # Full modelling grid INTERSECTION KMC boundary.
    #
    # Do not replace or regenerate the original grids.

    aligned_full = (
        full_grid
        .set_index("grid_id")
        .geometry
        .reindex(grid["grid_id"])
    )

    aligned_full = gpd.GeoSeries(
        aligned_full.to_numpy(),
        crs=grid.crs,
    )

    full_after_kmc_clip = gpd.GeoSeries(
        aligned_full
        .intersection(kmc_boundary)
        .to_numpy(),
        crs=grid.crs,
    )

    clipped_suitability = (
        grid.geometry
        .reset_index(drop=True)
    )

    difference_area = (
        clipped_suitability
        .symmetric_difference(
            full_after_kmc_clip,
            align=False,
        )
        .area
        .to_numpy(dtype=float)
    )

    max_difference = float(
        difference_area.max()
    )

    nontrivial_differences = int(
        (
            difference_area
            > GEOMETRY_MATCH_TOLERANCE_M2
        ).sum()
    )

    check(
        "Suitability matches modelling after KMC clipping",

        nontrivial_differences == 0,

        (
            f"Maximum difference: "
            f"{max_difference:.6f} m2; "
            f"cells above tolerance: "
            f"{nontrivial_differences}"
        ),
    )

    # ========================================================
    # 10. VALIDATE KMC AREA COVERAGE
    # ========================================================

    original_grid_area = (
        clipped_suitability.area
        .to_numpy(dtype=float)
    )

    inside_area = geometry_overlap_area(
        clipped_suitability,
        kmc_boundary,
    )

    outside_area = np.maximum(
        0.0,
        original_grid_area - inside_area,
    )

    check(
        "Every suitability grid intersects KMC",

        bool(
            (inside_area > 0).all()
        ),
    )

    check(
        "No material suitability geometry outside KMC",

        bool(
            (
                outside_area
                <= GEOMETRY_MATCH_TOLERANCE_M2
            ).all()
        ),

        (
            f"Maximum outside area: "
            f"{outside_area.max():.6f} m2"
        ),
    )

    check(
        "Sum of clipped grid areas matches KMC area",

        abs(
            float(inside_area.sum())
            - float(kmc_boundary.area)
        ) <= 1.0,

        (
            f"Grid: "
            f"{inside_area.sum()/10000:.4f} ha; "
            f"KMC: "
            f"{kmc_boundary.area/10000:.4f} ha"
        ),
    )

    stop_on_fail()

    print(
        "Verified grid IDs:",
        len(grid)
    )

    print(
        "Original full-vs-clipped differing cells:",
        previous_m1.get(
            "cells_above_tolerance"
        ),
    )

    print(
        "Maximum difference AFTER clipping:",
        round(max_difference, 6),
        "m2",
    )

    print(
        "Using the original Step 16B "
        "boundary-clipped grid."
    )

    # ========================================================
    # 11. LOAD PROVISIONAL PLANNING LAYERS
    # ========================================================

    print(
        "\n=== LOADING PROVISIONAL PLANNING LAYERS ==="
    )

    zones = validated_projected(
        gpd.read_file(
            ZONE_GPKG,
            layer="UDA_2024_ZONE_ESTIMATES",
        ),
        "Zone estimates",
    )

    unknown = validated_projected(
        gpd.read_file(
            SCREEN_GPKG,
            layer="UNKNOWN_ZONING_130",
        ),
        "Unknown zoning",
    )

    ambiguous = validated_projected(
        gpd.read_file(
            ZONE_GPKG,
            layer="QA_171_Ambiguous_Names",
        ),
        "Ambiguous names",
    )

    conservation = validated_projected(
        gpd.read_file(
            SCREEN_GPKG,
            layer="CONSERVATION_WATER_188",
        ),
        "Conservation/water",
    )

    highland = validated_projected(
        gpd.read_file(
            HIGHLAND_GEOJSON
        ),
        "Approximate highland",
    )

    check(
        "Planning feature counts match inventories",

        (
            len(zones),
            len(unknown),
            len(ambiguous),
            len(conservation),
            len(highland),
        ) == (
            508,
            130,
            171,
            188,
            2,
        ),
    )

    mandatory_fields = {
        "zone_id",
        "zone_name",
        "group_name",
    }

    check(
        "Zoning attribute fields available",

        mandatory_fields.issubset(
            zones.columns
        ),

        f"Columns: {list(zones.columns)}",
    )

    stop_on_fail()

    # ========================================================
    # 12. BUILD EXCLUSIVE ZONING CATEGORIES
    # ========================================================

    classified_polygon = (
        union_all(zones.geometry)
        .intersection(kmc_boundary)
    )

    unknown_polygon = (
        union_all(unknown.geometry)
        .intersection(kmc_boundary)
        .difference(classified_polygon)
    )

    unmapped_polygon = (
        kmc_boundary.difference(
            classified_polygon.union(
                unknown_polygon
            )
        )
    )

    # Additional review layers can overlap each other.

    ambiguous_polygon = (
        union_all(ambiguous.geometry)
        .intersection(kmc_boundary)
    )

    conservation_polygon = (
        union_all(conservation.geometry)
        .intersection(kmc_boundary)
    )

    highland_polygon = (
        union_all(highland.geometry)
        .intersection(kmc_boundary)
    )

    calculated_areas = {
        "classified_area_ha":
            classified_polygon.area / 10000,

        "unclassified_only_area_ha":
            unknown_polygon.area / 10000,

        "truly_unmapped_area_ha":
            unmapped_polygon.area / 10000,
    }

    coverage_matches = all(
        abs(
            float(value)
            - float(previous_j[key])
        ) < AREA_RECONCILIATION_HA

        for key, value
        in calculated_areas.items()
    )

    check(
        "Zoning areas reproduce Step 16C-J",

        coverage_matches,

        str(calculated_areas),
    )

    stop_on_fail()

    # ========================================================
    # 13. GRID-LEVEL SPATIAL INTERSECTIONS
    # ========================================================

    print(
        "\n=== COMPUTING GRID INTERSECTIONS ==="
    )

    polygons = {
        "classified": classified_polygon,
        "unclassified": unknown_polygon,
        "unmapped": unmapped_polygon,
        "ambiguous": ambiguous_polygon,
        "conservation_water": conservation_polygon,
        "highland_approx": highland_polygon,
    }

    overlap = {
        name: geometry_overlap_area(
            clipped_suitability,
            polygon,
        )

        for name, polygon in polygons.items()
    }

    partition_sum = (
        overlap["classified"]
        + overlap["unclassified"]
        + overlap["unmapped"]
    )

    max_partition_error = float(
        np.max(
            np.abs(
                partition_sum - inside_area
            ) / inside_area
        )
    )

    check(
        "Exclusive zoning categories partition every cell",

        max_partition_error <= 1e-5,

        (
            f"Maximum relative error: "
            f"{max_partition_error:.10f}"
        ),
    )

    for name, values in overlap.items():

        valid = bool(
            np.isfinite(values).all()
            and (values >= -1e-5).all()
            and (
                values <= inside_area + 1e-5
            ).all()
        )

        check(
            f"{name} intersections valid",
            valid,
        )

    stop_on_fail()

    # ========================================================
    # 14. BUILD OUTPUT GEODATAFRAME
    # ========================================================

    # Preserve Step 16B original clipped geometry.

    output = gpd.GeoDataFrame(
        {
            "grid_id": (
                grid["grid_id"]
                .astype(str)
                .to_numpy()
            )
        },

        geometry=clipped_suitability.to_numpy(),

        crs=grid.crs,
    )

    output["area_kmc_m2"] = (
        np.round(
            inside_area,
            4,
        )
    )

    output["grid_clip_pct"] = (
        np.round(
            100
            * inside_area
            / original_grid_area,
            4,
        )
    )

    for name, values in overlap.items():

        output[f"{name}_pct"] = (
            np.round(
                100 * values / inside_area,
                4,
            )
        )

    fractions = np.stack(
        [
            overlap["classified"] / inside_area,
            overlap["unclassified"] / inside_area,
            overlap["unmapped"] / inside_area,
        ],
        axis=1,
    )

    coverage_names = np.array([
        "CLASSIFIED",
        "UNCLASSIFIED",
        "UNMAPPED",
    ])

    dominant_index = fractions.argmax(
        axis=1
    )

    output["dominant_coverage"] = (
        coverage_names[dominant_index]
    )

    output["coverage_label"] = np.where(
        fractions.max(axis=1)
        >= PURE_CELL_FRACTION,

        coverage_names[dominant_index],

        "MIXED_COVERAGE",
    )

    output["zoning_legal_status"] = (
        "UNVERIFIED"
    )

    # ========================================================
    # 15. DOMINANT PROVISIONAL ZONING NAMES
    # ========================================================

    print(
        "\n=== IDENTIFYING DOMINANT "
        "PROVISIONAL ZONES ==="
    )

    attribute_names = [
        field
        for field in [
            "zone_id",
            "zone_name",
            "group_name",
            "confidence",
            "legal_status",
        ]
        if field in zones.columns
    ]

    zone_features = zones[
        attribute_names + ["geometry"]
    ].copy()

    zone_features["geometry"] = (
        zone_features.geometry.intersection(
            kmc_boundary
        )
    )

    zone_features = (
        zone_features.loc[
            ~zone_features.geometry.is_empty
        ]
        .reset_index(drop=True)
    )

    # First identify potential polygon intersections.
    # Then calculate their exact shared area.

    grid_join = gpd.GeoDataFrame(
        {
            "grid_id": output["grid_id"]
        },

        geometry=clipped_suitability.to_numpy(),

        crs=grid.crs,
    )

    joined = gpd.sjoin(
        grid_join,
        zone_features,
        how="inner",
        predicate="intersects",
    ).reset_index(drop=True)

    if len(joined) > 0:

        right_geometry = gpd.GeoSeries(
            zone_features.geometry.iloc[
                joined["index_right"]
                .to_numpy(dtype=int)
            ].to_numpy(),

            crs=grid.crs,
        )

        left_geometry = gpd.GeoSeries(
            joined.geometry.to_numpy(),
            crs=grid.crs,
        )

        joined["overlap_m2"] = (
            left_geometry.intersection(
                right_geometry,
                align=False,
            )
            .area
            .to_numpy(dtype=float)
        )

        joined = joined.loc[
            joined["overlap_m2"]
            > MIN_ZONE_INTERSECTION_M2
        ].copy()

    output["zone_piece_count"] = 0
    output["zone_class_count"] = 0

    for field in [
        "zone_id",
        "zone_name",
        "group_name",
        "confidence",
        "legal_status",
    ]:

        if field in zone_features.columns:

            output[f"main_zone_{field}"] = (
                "NOT_CLASSIFIED"
            )

    output["main_zone_pct"] = 0.0

    if not joined.empty:

        for field in attribute_names:

            joined[field] = (
                joined[field]
                .fillna("NOT_AVAILABLE")
                .astype(str)
            )

        piece_counts = joined.groupby(
            "grid_id"
        ).size()

        output["zone_piece_count"] = (
            output["grid_id"]
            .map(piece_counts)
            .fillna(0)
            .astype(int)
        )

        # If multiple polygons have the same zone
        # classification, combine their shared areas.

        zone_groups = joined.groupby(
            ["grid_id"] + attribute_names,
            dropna=False,
            as_index=False,
        )["overlap_m2"].sum()

        group_counts = zone_groups.groupby(
            "grid_id"
        ).size()

        output["zone_class_count"] = (
            output["grid_id"]
            .map(group_counts)
            .fillna(0)
            .astype(int)
        )

        zone_groups = zone_groups.sort_values(
            [
                "grid_id",
                "overlap_m2",
                "zone_id",
            ],

            ascending=[
                True,
                False,
                True,
            ],
        )

        dominant = (
            zone_groups
            .drop_duplicates("grid_id")
            .set_index("grid_id")
        )

        for field in attribute_names:

            output[f"main_zone_{field}"] = (
                output["grid_id"]
                .map(dominant[field])
                .fillna("NOT_CLASSIFIED")
            )

        largest_area = (
            output["grid_id"]
            .map(dominant["overlap_m2"])
            .fillna(0)
        )

        output["main_zone_pct"] = (
            np.round(
                100
                * largest_area.to_numpy(dtype=float)
                / inside_area,
                4,
            )
        )

    # ========================================================
    # 16. CREATE SCREENING REVIEW FLAGS
    # ========================================================

    have_unknown = (
        overlap["unclassified"]
        + overlap["unmapped"]
    ) > MIN_REVIEW_INTERSECTION_M2

    have_ambiguous = (
        overlap["ambiguous"]
        > MIN_REVIEW_INTERSECTION_M2
    )

    have_conservation = (
        overlap["conservation_water"]
        > MIN_REVIEW_INTERSECTION_M2
    )

    have_highland = (
        overlap["highland_approx"]
        > MIN_REVIEW_INTERSECTION_M2
    )

    review_flags = []

    for i in range(len(output)):

        flags = []

        if have_unknown[i]:

            flags.append(
                "UNKNOWN_OR_UNMAPPED"
            )

        if (
            output.at[i, "coverage_label"]
            == "MIXED_COVERAGE"
        ):

            flags.append(
                "MIXED_ZONING_COVERAGE"
            )

        if have_ambiguous[i]:

            flags.append(
                "AMBIGUOUS_ZONE_NAME"
            )

        if have_conservation[i]:

            flags.append(
                "CONSERVATION_WATER_REVIEW"
            )

        if have_highland[i]:

            flags.append(
                "APPROXIMATE_HIGHLAND_REVIEW"
            )

        review_flags.append(
            " | ".join(flags)
            if flags
            else "PROVISIONAL_CONTEXT_ONLY"
        )

    output["review_flags"] = review_flags

    output["review_needed"] = (
        output["review_flags"]
        != "PROVISIONAL_CONTEXT_ONLY"
    )

    # ========================================================
    # 17. FINAL QA
    # ========================================================

    check(
        "9240 unique output grid IDs",

        len(output) == EXPECTED_CELLS
        and output["grid_id"].is_unique,
    )

    check(
        "All zoning labels remain unverified",

        output["zoning_legal_status"]
        .eq("UNVERIFIED")
        .all(),
    )

    classified_cells = output.loc[
        overlap["classified"]
        > MIN_REVIEW_INTERSECTION_M2
    ]

    check(
        "Classified cells have provisional zone labels",

        bool(
            classified_cells[
                "main_zone_zone_name"
            ]
            .ne("NOT_CLASSIFIED")
            .all()
        ),
    )

    check(
        "Dominant zone share within classified share",

        bool(
            (
                output["main_zone_pct"]
                .to_numpy(dtype=float)

                <=

                output["classified_pct"]
                .to_numpy(dtype=float)
                + 0.001
            ).all()
        ),
    )

    check(
        "All source files unchanged",

        all(
            sha256(path)
            == source_digests[str(path)]

            for path in inputs
        ),
    )

    stop_on_fail()

    # ========================================================
    # 18. SAVE FINAL REPORTS
    # ========================================================

    groups = (
        output
        .groupby(
            "coverage_label",
            dropna=False,
        )
        .agg(
            grid_cells=(
                "grid_id",
                "size",
            ),

            clipped_grid_area_m2=(
                "area_kmc_m2",
                "sum",
            ),

            review_cells=(
                "review_needed",
                "sum",
            ),
        )
        .reset_index()
    )

    groups["clipped_grid_area_ha"] = (
        np.round(
            groups["clipped_grid_area_m2"]
            / 10000,
            4,
        )
    )

    groups = groups.drop(
        columns=[
            "clipped_grid_area_m2"
        ]
    )

    # Write a staging GeoPackage first.
    # Replace only this step's generated output
    # after the GIS file has been validated.

    temporary_gpkg = OUT_GPKG.with_name(
        OUT_GPKG.stem + "_staging.gpkg"
    )

    if temporary_gpkg.exists():

        temporary_gpkg.unlink()

    output.to_file(
        temporary_gpkg,
        layer="grid_provisional_screening",
        driver="GPKG",
        index=False,
    )

    saved_gpkg = gpd.read_file(
        temporary_gpkg
    )

    if len(saved_gpkg) != EXPECTED_CELLS:

        temporary_gpkg.unlink()

        raise RuntimeError(
            "Saved GIS file validation failed."
        )

    os.replace(
        temporary_gpkg,
        OUT_GPKG,
    )

    output.drop(
        columns="geometry"
    ).to_csv(
        OUT_CSV,
        index=False,
    )

    groups.to_csv(
        OUT_GROUPS,
        index=False,
    )

    qa = pd.DataFrame(
        checks
    )

    qa.to_csv(
        OUT_QA,
        index=False,
    )

    # ========================================================
    # 19. SAVE SUMMARY JSON
    # ========================================================

    summary = {
        "step": "16C-M",

        "qa_status": "PASS",

        "generated_utc": datetime.now(
            timezone.utc
        ).isoformat(),

        "grid_count": len(output),

        "source_suitability_layer": grid_layer,

        "source_modelling_layer": modelling_layer,

        "crs": f"EPSG:{EPSG}",

        "raw_csv_sha256":
            source_digests[str(RAW_CSV)],

        "suitability_gpkg_sha256":
            source_digests[str(SUITABILITY_GPKG)],

        "comparison_rule": (
            "Compare Step 16B clipped geometry "
            "with full modelling cells after "
            "clipping to KMC boundary."
        ),

        "maximum_clipped_grid_symmetric_difference_m2":
            max_difference,

        "cells_exceeding_grid_match_tolerance":
            nontrivial_differences,

        "max_zoning_partition_error_fraction":
            max_partition_error,

        "coverage_labels": {
            str(key): int(value)

            for key, value
            in output[
                "coverage_label"
            ].value_counts().items()
        },

        "ambiguous_review_cells": int(
            have_ambiguous.sum()
        ),

        "conservation_water_review_cells": int(
            have_conservation.sum()
        ),

        "approximate_highland_review_cells": int(
            have_highland.sum()
        ),

        "unknown_or_unmapped_review_cells": int(
            have_unknown.sum()
        ),

        "any_review_cells": int(
            output["review_needed"].sum()
        ),

        "qa_checks": len(checks),

        "qa_passed": int(
            qa["passed"].sum()
        ),

        "limitations": [
            (
                "Zoning is screenshot-derived, "
                "provisional, low confidence "
                "and legally unverified."
            ),

            (
                "The original Step 16B "
                "boundary-clipped grid is retained."
            ),

            (
                "Classified, unclassified and "
                "unmapped are exclusive categories."
            ),

            (
                "Conservation, water, ambiguous "
                "and highland review layers "
                "may overlap other categories."
            ),

            (
                "Review flags are not legal exclusions, "
                "construction approvals or suitability scores."
            ),

            (
                "Very small KMC boundary fragments "
                "are retained. Percentages use "
                "actual clipped-cell area."
            ),

            (
                "No AHP weights, final suitability "
                "index or official zoning rules applied."
            ),
        ],

        "outputs": [
            path.name
            for path in [
                OUT_CSV,
                OUT_GPKG,
                OUT_GROUPS,
                OUT_QA,
                OUT_SUMMARY,
            ]
        ],
    }

    OUT_SUMMARY.write_text(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    # ========================================================
    # 20. PRINT TERMINAL RESULTS
    # ========================================================

    print(
        "\n=== GRID SCREENING CATEGORIES ==="
    )

    print(
        groups.to_string(
            index=False
        )
    )

    print(
        "\n=== REVIEW INDICATORS ==="
    )

    print(
        "Unknown/unmapped:",
        summary["unknown_or_unmapped_review_cells"],
    )

    print(
        "Ambiguous zone name:",
        summary["ambiguous_review_cells"],
    )

    print(
        "Conservation/water review:",
        summary["conservation_water_review_cells"],
    )

    print(
        "Approximate highland review:",
        summary["approximate_highland_review_cells"],
    )

    print(
        "Any review flag:",
        summary["any_review_cells"],
    )

    print(
        "\n=== QA RESULTS ==="
    )

    for row in checks:

        print(
            "PASS:" if row["passed"] else "FAIL:",
            row["check"],
        )

    print(
        f"\nOverall QA: PASS "
        f"({len(checks)}/{len(checks)})"
    )

    print(
        "\n=== GENERATED REPORTS ==="
    )

    for path in [
        OUT_CSV,
        OUT_GPKG,
        OUT_GROUPS,
        OUT_QA,
        OUT_SUMMARY,
    ]:

        print(
            path.relative_to(ROOT)
        )

    print(
        "\nPlanning data status: "
        "PROVISIONAL / LEGALLY UNVERIFIED"
    )

    print(
        "No AHP weights or final suitability "
        "scores calculated."
    )

    print("=" * 95)


# ============================================================
# 21. EXECUTE
# ============================================================

if __name__ == "__main__":

    main()
