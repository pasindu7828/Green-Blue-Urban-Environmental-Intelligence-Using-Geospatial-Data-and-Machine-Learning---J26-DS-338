from pathlib import Path
import html
import json
import re

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely import make_valid
from shapely.ops import unary_union


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_BOUNDARY_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "boundaries"
)

MAP_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "maps"
    / "part2_candidate_wetland"
)

REPORT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "reports"
    / "part2_candidate_wetland"
)


# ============================================================
# INPUTS
# ============================================================

GN_KML = (
    RAW_BOUNDARY_DIR
    / "Kaduwela TO Area.kml"
)

GN_LAYER = (
    "GN Boundary Colombo District (#2)"
)

PRIMARY_CANDIDATES = (
    MAP_DIR
    / "20_primary_candidate_layer_utm_repaired.geojson"
)


# ============================================================
# OUTPUTS
# ============================================================

INTERSECTION_CSV = (
    REPORT_DIR
    / "22_primary_candidate_gn_intersections.csv"
)

GN_SUMMARY_CSV = (
    REPORT_DIR
    / "22_gn_primary_candidate_summary.csv"
)

INTERSECTION_UTM = (
    MAP_DIR
    / "22_primary_candidate_gn_intersections_utm.geojson"
)

GN_SUMMARY_UTM = (
    MAP_DIR
    / "22_gn_primary_candidate_summary_utm.geojson"
)

GN_SUMMARY_WGS84 = (
    MAP_DIR
    / "22_gn_primary_candidate_summary_wgs84.geojson"
)

AUDIT_OUTPUT = (
    REPORT_DIR
    / "22_gn_candidate_summary_audit.json"
)


# ============================================================
# HELPERS
# ============================================================

def extract_description_field(
    description,
    field_name,
):
    """
    Extract a value from the HTML table stored inside the
    KML description field.
    """

    if (
        description is None
        or
        pd.isna(description)
    ):
        return None

    pattern = (
        rf"<td>\s*{re.escape(field_name)}\s*</td>"
        rf"\s*<td>\s*(.*?)\s*</td>"
    )

    match = re.search(
        pattern,
        str(description),
        flags=re.IGNORECASE | re.DOTALL,
    )

    if not match:
        return None

    value = html.unescape(
        match.group(1)
    )

    value = re.sub(
        r"<[^>]+>",
        "",
        value,
    )

    value = value.strip()

    if value.lower() in {
        "",
        "<null>",
        "null",
        "none",
        "nan",
    }:
        return None

    return value


def repair_geometry(
    geometry,
):

    if (
        geometry is None
        or
        geometry.is_empty
    ):
        return geometry

    if not geometry.is_valid:
        geometry = make_valid(
            geometry
        )

    if (
        geometry is not None
        and
        not geometry.is_empty
        and
        not geometry.is_valid
    ):
        geometry = geometry.buffer(
            0
        )

    return geometry


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "\n========================================"
    )

    print(
        "PART 2 GN-WISE CANDIDATE REPORTING"
    )

    print(
        "========================================"
    )

    print(
        "\nGN source layer:"
    )

    print(
        GN_LAYER
    )

    # ========================================================
    # 1. LOAD GN BOUNDARIES
    # ========================================================

    gn = gpd.read_file(
        GN_KML,
        layer=GN_LAYER,
    )

    print(
        f"\nGN polygons loaded: "
        f"{len(gn)}"
    )

    print(
        f"Original GN CRS: "
        f"{gn.crs}"
    )

    if len(gn) != 56:

        raise RuntimeError(
            f"Expected 56 GN polygons, "
            f"found {len(gn)}."
        )

    # ========================================================
    # 2. EXTRACT GN ATTRIBUTES FROM KML DESCRIPTION
    # ========================================================

    gn[
        "gn_name_description"
    ] = gn[
        "description"
    ].apply(
        lambda value:
            extract_description_field(
                value,
                "GND_N",
            )
    )

    gn[
        "gn_number"
    ] = gn[
        "description"
    ].apply(
        lambda value:
            extract_description_field(
                value,
                "GND_NO",
            )
    )

    gn[
        "admin_code"
    ] = gn[
        "description"
    ].apply(
        lambda value:
            extract_description_field(
                value,
                "Admin_Code",
            )
    )

    gn[
        "ds_division"
    ] = gn[
        "description"
    ].apply(
        lambda value:
            extract_description_field(
                value,
                "DSD_N",
            )
    )

    # Prefer GND_N from description.
    # Fall back to KML Name if needed.

    gn[
        "gn_name"
    ] = gn[
        "gn_name_description"
    ]

    missing_name = (
        gn[
            "gn_name"
        ]
        .isna()
    )

    gn.loc[
        missing_name,
        "gn_name",
    ] = gn.loc[
        missing_name,
        "Name",
    ]

    # Build stable GN identifier.

    gn[
        "gn_id"
    ] = gn[
        "admin_code"
    ]

    missing_id = (
        gn[
            "gn_id"
        ]
        .isna()
    )

    gn.loc[
        missing_id,
        "gn_id",
    ] = (
        "GN_"
        +
        gn.loc[
            missing_id
        ]
        .index
        .astype(str)
        .str.zfill(3)
    )

    if (
        gn[
            "gn_id"
        ]
        .duplicated()
        .any()
    ):

        raise RuntimeError(
            "Duplicate GN identifiers detected."
        )

    print(
        "\nGN attribute extraction:"
    )

    print(
        f"GN names available: "
        f"{gn['gn_name'].notna().sum()} / 56"
    )

    print(
        f"GN numbers available: "
        f"{gn['gn_number'].notna().sum()} / 56"
    )

    print(
        f"Admin codes available: "
        f"{gn['admin_code'].notna().sum()} / 56"
    )

    # ========================================================
    # 3. GEOMETRY QA + PROJECT TO UTM 44N
    # ========================================================

    gn[
        "geometry"
    ] = gn.geometry.apply(
        repair_geometry
    )

    if (
        not gn.geometry.is_valid.all()
    ):

        raise RuntimeError(
            "GN geometry remains invalid."
        )

    gn_utm = gn.to_crs(
        epsg=32644
    )

    gn_utm[
        "gn_area_ha"
    ] = (
        gn_utm.geometry.area
        /
        10000.0
    )

    # ========================================================
    # 4. LOAD FINAL PRIMARY CANDIDATES
    # ========================================================

    candidates = gpd.read_file(
        PRIMARY_CANDIDATES
    )

    print(
        f"\nPrimary candidate polygons: "
        f"{len(candidates)}"
    )

    print(
        f"Candidate CRS: "
        f"{candidates.crs}"
    )

    if len(candidates) != 48:

        raise RuntimeError(
            "Expected 48 primary candidate polygons."
        )

    if (
        candidates.crs.to_epsg()
        !=
        32644
    ):

        raise RuntimeError(
            "Primary candidate layer must use EPSG:32644."
        )

    if (
        not candidates.geometry.is_valid.all()
    ):

        raise RuntimeError(
            "Primary candidate geometry is invalid."
        )

    candidates[
        "candidate_geometry_area_ha"
    ] = (
        candidates.geometry.area
        /
        10000.0
    )

    total_primary_area_ha = float(
        candidates[
            "candidate_geometry_area_ha"
        ].sum()
    )

    print(
        f"Primary candidate geometry area: "
        f"{total_primary_area_ha:.4f} ha"
    )

    # ========================================================
    # 5. GN PARTITION QA
    # ========================================================

    gn_union = unary_union(
        gn_utm.geometry.tolist()
    )

    gn_area_sum_ha = float(
        gn_utm[
            "gn_area_ha"
        ].sum()
    )

    gn_union_area_ha = float(
        gn_union.area
        /
        10000.0
    )

    apparent_overlap_ha = (
        gn_area_sum_ha
        -
        gn_union_area_ha
    )

    print(
        "\nGN boundary QA:"
    )

    print(
        f"Sum of GN polygon areas: "
        f"{gn_area_sum_ha:.4f} ha"
    )

    print(
        f"GN union area: "
        f"{gn_union_area_ha:.4f} ha"
    )

    print(
        f"Apparent GN overlap area: "
        f"{apparent_overlap_ha:.8f} ha"
    )

    # ========================================================
    # 6. CANDIDATE x GN INTERSECTION
    # ========================================================

    candidate_fields = [
        "candidate_id",
        "screening_priority",
        "candidate_reporting_status",
        "manual_qa_reviewed",
        "qa_final_action",
        "surface_review_flag",
        "area_ha",
        "mean_probability",
        "fraction_ge_0p80",
        "candidate_geometry_area_ha",
        "geometry",
    ]

    gn_fields = [
        "gn_id",
        "gn_name",
        "gn_number",
        "admin_code",
        "gn_area_ha",
        "geometry",
    ]

    intersections = gpd.overlay(
        candidates[
            candidate_fields
        ],
        gn_utm[
            gn_fields
        ],
        how="intersection",
        keep_geom_type=False,
    )

    # Keep polygonal area intersections only.

    intersections = (
        intersections[
            intersections.geometry.geom_type.isin(
                [
                    "Polygon",
                    "MultiPolygon",
                ]
            )
        ]
        .copy()
    )

    intersections[
        "intersection_area_ha"
    ] = (
        intersections.geometry.area
        /
        10000.0
    )

    intersections = (
        intersections[
            intersections[
                "intersection_area_ha"
            ]
            > 0
        ]
        .copy()
    )

    print(
        f"\nCandidate-GN intersection records: "
        f"{len(intersections)}"
    )

    # ========================================================
    # 7. CANDIDATE FRACTION WITHIN EACH GN
    # ========================================================

    intersections[
        "candidate_fraction_in_gn_pct"
    ] = (
        100.0
        *
        intersections[
            "intersection_area_ha"
        ]
        /
        intersections[
            "candidate_geometry_area_ha"
        ]
    )

    intersections[
        "gn_area_covered_pct"
    ] = (
        100.0
        *
        intersections[
            "intersection_area_ha"
        ]
        /
        intersections[
            "gn_area_ha"
        ]
    )

    # ========================================================
    # 8. DOMINANT GN FOR EACH CANDIDATE
    #
    # Candidate can intersect several GNs.
    # Dominant GN = GN containing the largest candidate area.
    # ========================================================

    dominant = (
        intersections
        .sort_values(
            [
                "candidate_id",
                "intersection_area_ha",
                "gn_id",
            ],
            ascending=[
                True,
                False,
                True,
            ],
        )
        .drop_duplicates(
            subset=[
                "candidate_id"
            ],
            keep="first",
        )
        [
            [
                "candidate_id",
                "gn_id",
            ]
        ]
        .rename(
            columns={
                "gn_id":
                    "dominant_gn_id"
            }
        )
    )

    intersections = (
        intersections.merge(
            dominant,
            on="candidate_id",
            how="left",
            validate="many_to_one",
        )
    )

    intersections[
        "is_dominant_gn"
    ] = (
        intersections[
            "gn_id"
        ]
        ==
        intersections[
            "dominant_gn_id"
        ]
    )

    # ========================================================
    # 9. COVERAGE QA
    # ========================================================

    candidate_ids_with_gn = set(
        intersections[
            "candidate_id"
        ]
    )

    all_candidate_ids = set(
        candidates[
            "candidate_id"
        ]
    )

    candidates_without_gn = sorted(
        all_candidate_ids
        -
        candidate_ids_with_gn
    )

    total_intersection_area_ha = float(
        intersections[
            "intersection_area_ha"
        ].sum()
    )

    outside_area_ha = 0.0

    for geometry in (
        candidates.geometry
    ):

        outside_piece = (
            geometry.difference(
                gn_union
            )
        )

        if (
            outside_piece is not None
            and
            not outside_piece.is_empty
        ):

            outside_area_ha += (
                outside_piece.area
                /
                10000.0
            )

    outside_percentage = (
        100.0
        *
        outside_area_ha
        /
        total_primary_area_ha
        if total_primary_area_ha > 0
        else 0.0
    )

    print(
        "\nCandidate/GN coverage QA:"
    )

    print(
        f"Candidates with at least one GN intersection: "
        f"{len(candidate_ids_with_gn)} / 48"
    )

    print(
        f"Candidate area outside GN union: "
        f"{outside_area_ha:.6f} ha"
    )

    print(
        f"Outside percentage: "
        f"{outside_percentage:.6f}%"
    )

    if candidates_without_gn:

        print(
            "Candidates without GN intersection:"
        )

        print(
            candidates_without_gn
        )

    # ========================================================
    # 10. BUILD GN SUMMARY
    # ========================================================

    summary = (
        gn_utm[
            [
                "gn_id",
                "gn_name",
                "gn_number",
                "admin_code",
                "gn_area_ha",
            ]
        ]
        .copy()
    )

    # --------------------------------------------------------
    # Total intersecting candidate count
    # --------------------------------------------------------

    intersect_count = (
        intersections.groupby(
            "gn_id"
        )[
            "candidate_id"
        ]
        .nunique()
        .rename(
            "intersecting_primary_candidate_count"
        )
    )

    summary = summary.merge(
        intersect_count,
        on="gn_id",
        how="left",
    )

    # --------------------------------------------------------
    # Total candidate area in each GN
    # --------------------------------------------------------

    area_total = (
        intersections.groupby(
            "gn_id"
        )[
            "intersection_area_ha"
        ]
        .sum()
        .rename(
            "primary_candidate_area_ha"
        )
    )

    summary = summary.merge(
        area_total,
        on="gn_id",
        how="left",
    )

    # --------------------------------------------------------
    # Priority-specific count + area
    # --------------------------------------------------------

    for priority in [
        "High",
        "Medium",
        "Review",
    ]:

        subset = (
            intersections[
                intersections[
                    "screening_priority"
                ]
                ==
                priority
            ]
        )

        count_series = (
            subset.groupby(
                "gn_id"
            )[
                "candidate_id"
            ]
            .nunique()
            .rename(
                f"{priority.lower()}_intersecting_count"
            )
        )

        area_series = (
            subset.groupby(
                "gn_id"
            )[
                "intersection_area_ha"
            ]
            .sum()
            .rename(
                f"{priority.lower()}_candidate_area_ha"
            )
        )

        summary = summary.merge(
            count_series,
            on="gn_id",
            how="left",
        )

        summary = summary.merge(
            area_series,
            on="gn_id",
            how="left",
        )

    # --------------------------------------------------------
    # Dominant-GN count
    #
    # This counts every candidate once only.
    # --------------------------------------------------------

    dominant_intersections = (
        intersections[
            intersections[
                "is_dominant_gn"
            ]
        ]
        .copy()
    )

    dominant_count = (
        dominant_intersections.groupby(
            "gn_id"
        )[
            "candidate_id"
        ]
        .nunique()
        .rename(
            "dominant_primary_candidate_count"
        )
    )

    summary = summary.merge(
        dominant_count,
        on="gn_id",
        how="left",
    )

    # --------------------------------------------------------
    # Area-weighted mean probability
    # --------------------------------------------------------

    intersections[
        "_probability_area_product"
    ] = (
        intersections[
            "mean_probability"
        ]
        *
        intersections[
            "intersection_area_ha"
        ]
    )

    weighted_prob = (
        intersections.groupby(
            "gn_id"
        )
        .agg(
            probability_area_sum=(
                "_probability_area_product",
                "sum",
            ),

            probability_area_denominator=(
                "intersection_area_ha",
                "sum",
            ),
        )
    )

    weighted_prob[
        "area_weighted_mean_probability"
    ] = (
        weighted_prob[
            "probability_area_sum"
        ]
        /
        weighted_prob[
            "probability_area_denominator"
        ]
    )

    summary = summary.merge(
        weighted_prob[
            [
                "area_weighted_mean_probability"
            ]
        ],
        on="gn_id",
        how="left",
    )

    # ========================================================
    # 11. FILL ZERO-CANDIDATE GNs
    # ========================================================

    count_columns = [
        "intersecting_primary_candidate_count",
        "dominant_primary_candidate_count",
        "high_intersecting_count",
        "medium_intersecting_count",
        "review_intersecting_count",
    ]

    area_columns = [
        "primary_candidate_area_ha",
        "high_candidate_area_ha",
        "medium_candidate_area_ha",
        "review_candidate_area_ha",
    ]

    for column in count_columns:

        if column not in summary.columns:
            summary[
                column
            ] = 0

        summary[
            column
        ] = (
            summary[
                column
            ]
            .fillna(0)
            .astype(int)
        )

    for column in area_columns:

        if column not in summary.columns:
            summary[
                column
            ] = 0.0

        summary[
            column
        ] = (
            summary[
                column
            ]
            .fillna(0.0)
        )

    summary[
        "primary_candidate_area_pct_of_gn"
    ] = (
        100.0
        *
        summary[
            "primary_candidate_area_ha"
        ]
        /
        summary[
            "gn_area_ha"
        ]
    )

    summary[
        "has_primary_candidate"
    ] = (
        summary[
            "intersecting_primary_candidate_count"
        ]
        >
        0
    )

    # ========================================================
    # 12. SORT GN SUMMARY
    # ========================================================

    summary = (
        summary.sort_values(
            [
                "primary_candidate_area_ha",
                "intersecting_primary_candidate_count",
                "gn_name",
            ],
            ascending=[
                False,
                False,
                True,
            ],
        )
        .reset_index(
            drop=True
        )
    )

    summary[
        "candidate_area_rank"
    ] = np.arange(
        1,
        len(summary) + 1,
    )

    # ========================================================
    # 13. SAVE CSV INTERSECTION TABLE
    # ========================================================

    intersection_csv = (
        intersections.drop(
            columns="geometry"
        )
        .copy()
    )

    intersection_csv.to_csv(
        INTERSECTION_CSV,
        index=False,
    )

    summary.to_csv(
        GN_SUMMARY_CSV,
        index=False,
    )

    # ========================================================
    # 14. SAVE SPATIAL INTERSECTION LAYER
    # ========================================================

    intersections.drop(
        columns=[
            "_probability_area_product"
        ],
        errors="ignore",
    ).to_file(
        INTERSECTION_UTM,
        driver="GeoJSON",
    )

    # ========================================================
    # 15. BUILD GN SPATIAL SUMMARY
    # ========================================================

    gn_spatial = (
        gn_utm[
            [
                "gn_id",
                "geometry",
            ]
        ]
        .merge(
            summary,
            on="gn_id",
            how="left",
            validate="one_to_one",
        )
    )

    gn_spatial.to_file(
        GN_SUMMARY_UTM,
        driver="GeoJSON",
    )

    gn_spatial.to_crs(
        epsg=4326
    ).to_file(
        GN_SUMMARY_WGS84,
        driver="GeoJSON",
    )

    # ========================================================
    # 16. FINAL SUMMARY STATS
    # ========================================================

    gn_with_candidates = int(
        summary[
            "has_primary_candidate"
        ].sum()
    )

    gn_without_candidates = (
        56
        -
        gn_with_candidates
    )

    total_dominant_count = int(
        summary[
            "dominant_primary_candidate_count"
        ].sum()
    )

    # ========================================================
    # 17. AUDIT
    # ========================================================

    audit = {
        "gn_source_file":
            str(
                GN_KML
            ),

        "gn_source_layer":
            GN_LAYER,

        "gn_polygon_count":
            int(
                len(
                    gn_utm
                )
            ),

        "primary_candidate_count":
            int(
                len(
                    candidates
                )
            ),

        "candidate_gn_intersection_records":
            int(
                len(
                    intersections
                )
            ),

        "candidates_with_gn_intersection":
            int(
                len(
                    candidate_ids_with_gn
                )
            ),

        "candidates_without_gn_intersection":
            candidates_without_gn,

        "gn_with_primary_candidates":
            gn_with_candidates,

        "gn_without_primary_candidates":
            gn_without_candidates,

        "dominant_gn_candidate_count_total":
            total_dominant_count,

        "total_primary_candidate_geometry_area_ha":
            total_primary_area_ha,

        "total_intersection_area_ha":
            total_intersection_area_ha,

        "candidate_area_outside_gn_union_ha":
            outside_area_ha,

        "candidate_area_outside_gn_union_percentage":
            outside_percentage,

        "gn_polygon_area_sum_ha":
            gn_area_sum_ha,

        "gn_union_area_ha":
            gn_union_area_ha,

        "apparent_gn_overlap_area_ha":
            apparent_overlap_ha,

        "gn_assignment_method":
            (
                "Polygon intersection. "
                "Candidate area is split across GN divisions "
                "when a candidate crosses a GN boundary."
            ),

        "dominant_gn_definition":
            (
                "GN containing the largest intersected area "
                "of a candidate polygon."
            ),

        "important_interpretation":
            (
                "Primary candidates are research screening "
                "outputs and not officially confirmed wetlands."
            ),
    }

    with open(
        AUDIT_OUTPUT,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            audit,
            file,
            indent=4,
        )

    # ========================================================
    # TERMINAL REPORT
    # ========================================================

    print(
        "\n========================================"
    )

    print(
        "GN-WISE PRIMARY CANDIDATE SUMMARY"
    )

    print(
        "========================================"
    )

    display_columns = [
        "candidate_area_rank",
        "gn_name",
        "gn_number",
        "intersecting_primary_candidate_count",
        "dominant_primary_candidate_count",
        "high_intersecting_count",
        "medium_intersecting_count",
        "review_intersecting_count",
        "primary_candidate_area_ha",
        "primary_candidate_area_pct_of_gn",
        "area_weighted_mean_probability",
    ]

    print(
        summary[
            display_columns
        ]
        .to_string(
            index=False
        )
    )

    print(
        "\n========================================"
    )

    print(
        "GN REPORTING TOTALS"
    )

    print(
        "========================================"
    )

    print(
        f"GN divisions: "
        f"{len(summary)}"
    )

    print(
        f"GNs containing primary candidates: "
        f"{gn_with_candidates}"
    )

    print(
        f"GNs without primary candidates: "
        f"{gn_without_candidates}"
    )

    print(
        f"Unique primary candidates by dominant GN: "
        f"{total_dominant_count}"
    )

    print(
        f"Candidate area outside GN union: "
        f"{outside_area_ha:.6f} ha "
        f"({outside_percentage:.6f}%)"
    )

    print(
        "\nTop 10 GNs by primary candidate area:"
    )

    print(
        summary[
            display_columns
        ]
        .head(10)
        .to_string(
            index=False
        )
    )

    print(
        "\nOutputs:"
    )

    print(
        INTERSECTION_CSV
    )

    print(
        GN_SUMMARY_CSV
    )

    print(
        INTERSECTION_UTM
    )

    print(
        GN_SUMMARY_UTM
    )

    print(
        GN_SUMMARY_WGS84
    )

    print(
        AUDIT_OUTPUT
    )

    print(
        "\n========================================"
    )

    print(
        "GN-WISE CANDIDATE REPORTING: COMPLETE"
    )

    print(
        "========================================"
    )

    print(
        "\nIMPORTANT:"
        "\nCandidate counts based on intersection can "
        "count one candidate in more than one GN."
        "\nThe dominant-GN count assigns each candidate "
        "to exactly one GN for unique-count reporting."
        "\nCandidate area is allocated spatially using "
        "the actual GN intersection."
    )


if __name__ == "__main__":
    main()