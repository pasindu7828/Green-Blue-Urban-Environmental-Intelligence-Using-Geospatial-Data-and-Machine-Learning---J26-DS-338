from pathlib import Path

import pandas as pd
import geopandas as gpd
from shapely import make_valid


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_BOUNDARY_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "boundaries"
)

PROCESSED_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "part1_historical_change"
)

GN_KML = (
    RAW_BOUNDARY_DIR
    / "Kaduwela TO Area.kml"
)

WETLAND_INPUT = (
    PROCESSED_DIR
    / "wetland_latest_status_wgs84.geojson"
)

OVERLAP_OUTPUT = (
    PROCESSED_DIR
    / "wetland_gn_overlaps.csv"
)

WETLAND_GN_OUTPUT = (
    PROCESSED_DIR
    / "wetland_gn_summary.csv"
)

WETLAND_MAP_OUTPUT = (
    PROCESSED_DIR
    / "wetland_latest_status_gn_wgs84.geojson"
)

GN_SUMMARY_OUTPUT = (
    PROCESSED_DIR
    / "gn_wetland_summary.csv"
)


# ============================================================
# SETTINGS
# ============================================================

GN_LAYER = (
    "GN Boundary Colombo District (#2)"
)

METRIC_CRS = "EPSG:32644"


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n========================================")
    print("WETLAND TO GN DIVISION ASSIGNMENT")
    print("========================================")

    # --------------------------------------------------------
    # LOAD DATA
    # --------------------------------------------------------

    wetlands = gpd.read_file(
        WETLAND_INPUT
    )

    gn = gpd.read_file(
        GN_KML,
        layer=GN_LAYER,
    )

    print(
        f"Latest wetlands loaded: "
        f"{len(wetlands)}"
    )

    print(
        f"GN divisions loaded: "
        f"{len(gn)}"
    )

    # --------------------------------------------------------
    # KEEP REQUIRED GN FIELDS
    # --------------------------------------------------------

    gn = gn[
        [
            "Name",
            "geometry",
        ]
    ].copy()

    gn = gn.rename(
        columns={
            "Name": "gn_name"
        }
    )

    # --------------------------------------------------------
    # FIX GEOMETRIES IF REQUIRED
    # --------------------------------------------------------

    wetlands["geometry"] = (
        wetlands["geometry"]
        .apply(make_valid)
    )

    gn["geometry"] = (
        gn["geometry"]
        .apply(make_valid)
    )

    # --------------------------------------------------------
    # CONVERT TO METRIC CRS FOR AREA CALCULATION
    # --------------------------------------------------------

    wetlands_utm = wetlands.to_crs(
        METRIC_CRS
    )

    gn_utm = gn.to_crs(
        METRIC_CRS
    )

    # --------------------------------------------------------
    # CALCULATE LATEST WETLAND GEOMETRY AREA
    # --------------------------------------------------------

    wetland_area = (
        wetlands_utm[
            [
                "wetland_id",
                "geometry",
            ]
        ]
        .copy()
    )

    wetland_area[
        "wetland_geometry_area_ha"
    ] = (
        wetland_area.geometry.area
        / 10000.0
    )

    # --------------------------------------------------------
    # INTERSECT WETLANDS WITH GN DIVISIONS
    # --------------------------------------------------------

    intersections = gpd.overlay(
        wetland_area,
        gn_utm,
        how="intersection",
        keep_geom_type=False,
    )

    # Keep only area geometries.
    intersections = intersections[
        intersections.geometry.area > 0
    ].copy()

    intersections[
        "overlap_area_ha"
    ] = (
        intersections.geometry.area
        / 10000.0
    )

    intersections[
        "overlap_percent"
    ] = (
        intersections[
            "overlap_area_ha"
        ]
        /
        intersections[
            "wetland_geometry_area_ha"
        ]
        * 100.0
    )

    # --------------------------------------------------------
    # SORT EACH WETLAND FROM LARGEST GN OVERLAP
    # --------------------------------------------------------

    intersections = (
        intersections
        .sort_values(
            [
                "wetland_id",
                "overlap_area_ha",
            ],
            ascending=[
                True,
                False,
            ],
        )
        .reset_index(drop=True)
    )

    # --------------------------------------------------------
    # SAVE DETAILED OVERLAPS
    # --------------------------------------------------------

    overlap_table = intersections[
        [
            "wetland_id",
            "gn_name",
            "wetland_geometry_area_ha",
            "overlap_area_ha",
            "overlap_percent",
        ]
    ].copy()

    overlap_table.to_csv(
        OVERLAP_OUTPUT,
        index=False,
    )

    # --------------------------------------------------------
    # BUILD ONE SUMMARY ROW PER WETLAND
    # --------------------------------------------------------

    summary_rows = []

    for wetland_id, group in overlap_table.groupby(
        "wetland_id"
    ):

        group = (
            group
            .sort_values(
                "overlap_area_ha",
                ascending=False,
            )
            .reset_index(drop=True)
        )

        primary = group.iloc[0]

        total_coverage = (
            group[
                "overlap_percent"
            ].sum()
        )

        all_gns = " | ".join(
            group["gn_name"]
            .astype(str)
            .tolist()
        )

        summary_rows.append(
            {
                "wetland_id":
                    wetland_id,

                "primary_gn":
                    primary[
                        "gn_name"
                    ],

                "primary_gn_overlap_ha":
                    primary[
                        "overlap_area_ha"
                    ],

                "primary_gn_overlap_percent":
                    primary[
                        "overlap_percent"
                    ],

                "gn_count":
                    len(group),

                "all_gn_divisions":
                    all_gns,

                "gn_coverage_percent":
                    total_coverage,
            }
        )

    wetland_summary = pd.DataFrame(
        summary_rows
    )

    # --------------------------------------------------------
    # CHECK FOR WETLANDS WITH NO GN INTERSECTION
    # --------------------------------------------------------

    all_wetland_ids = set(
        wetlands["wetland_id"]
    )

    matched_wetland_ids = set(
        wetland_summary["wetland_id"]
    )

    unmatched_ids = sorted(
        all_wetland_ids
        - matched_wetland_ids
    )

    if unmatched_ids:

        print(
            "\nWARNING - wetlands without GN intersection:"
        )

        for wetland_id in unmatched_ids:
            print(
                f" - {wetland_id}"
            )

    # --------------------------------------------------------
    # SAVE WETLAND-GN SUMMARY
    # --------------------------------------------------------

    wetland_summary.to_csv(
        WETLAND_GN_OUTPUT,
        index=False,
    )

    # --------------------------------------------------------
    # ATTACH GN INFORMATION TO FINAL WETLAND MAP
    # --------------------------------------------------------

    wetlands_with_gn = wetlands.merge(
        wetland_summary,
        on="wetland_id",
        how="left",
        validate="one_to_one",
    )

    wetlands_with_gn.to_file(
        WETLAND_MAP_OUTPUT,
        driver="GeoJSON",
    )

    # --------------------------------------------------------
    # GN-WISE WETLAND SUMMARY
    #
    # Uses actual overlapping area, so a wetland crossing
    # two GN divisions contributes area to both.
    # --------------------------------------------------------

    gn_summary = (
        overlap_table
        .groupby(
            "gn_name"
        )
        .agg(
            known_wetland_count=(
                "wetland_id",
                "nunique",
            ),
            latest_wetland_area_ha=(
                "overlap_area_ha",
                "sum",
            ),
        )
        .reset_index()
    )

    wetland_names = (
        overlap_table
        .groupby(
            "gn_name"
        )[
            "wetland_id"
        ]
        .apply(
            lambda values:
                " | ".join(
                    sorted(
                        set(values)
                    )
                )
        )
        .reset_index(
            name="wetland_ids"
        )
    )

    gn_summary = gn_summary.merge(
        wetland_names,
        on="gn_name",
        how="left",
    )

    gn_summary.to_csv(
        GN_SUMMARY_OUTPUT,
        index=False,
    )

    # --------------------------------------------------------
    # TERMINAL REPORT
    # --------------------------------------------------------

    print("\nGN assignment summary:")

    print(
        f"Wetlands with GN match: "
        f"{wetland_summary['wetland_id'].nunique()}"
    )

    print(
        f"Wetlands without GN match: "
        f"{len(unmatched_ids)}"
    )

    print(
        f"Wetlands crossing multiple GN divisions: "
        f"{(wetland_summary['gn_count'] > 1).sum()}"
    )

    print(
        f"GN divisions containing at least one "
        f"known wetland: "
        f"{gn_summary['gn_name'].nunique()}"
    )

    print(
        "\nGN coverage QA:"
    )

    print(
        wetland_summary[
            "gn_coverage_percent"
        ]
        .describe()
        .to_string()
    )

    print(
        "\nWetland → GN assignments:"
    )

    print(
        wetland_summary[
            [
                "wetland_id",
                "primary_gn",
                "primary_gn_overlap_percent",
                "gn_count",
            ]
        ]
        .to_string(
            index=False
        )
    )

    print("\nOutputs:")

    print(OVERLAP_OUTPUT)
    print(WETLAND_GN_OUTPUT)
    print(WETLAND_MAP_OUTPUT)
    print(GN_SUMMARY_OUTPUT)

    print(
        "\nNOTE:"
        "\nPrimary GN = GN division containing the "
        "largest proportion of the latest wetland polygon."
        "\nWetlands crossing GN boundaries retain all "
        "overlapping GN divisions."
    )


if __name__ == "__main__":
    main()