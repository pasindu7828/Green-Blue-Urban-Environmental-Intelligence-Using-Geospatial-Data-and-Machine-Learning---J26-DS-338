from pathlib import Path

import pandas as pd
import geopandas as gpd

from shapely import make_valid
from shapely.geometry import (
    Polygon,
    MultiPolygon,
    GeometryCollection,
)
from shapely.ops import unary_union


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PROCESSED_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "part1_historical_change"
)

INTERIM_DIR = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "part1_historical_change"
)

PAIR_INPUT = (
    PROCESSED_DIR
    / "historical_wetland_change_pairs_quality.csv"
)

OBSERVATION_INPUT = (
    PROCESSED_DIR
    / "wetland_obs_rainfall_utm44n.gpkg"
)

METRICS_OUTPUT = (
    PROCESSED_DIR
    / "trusted_spatial_change_metrics.csv"
)

ZONE_OUTPUT = (
    PROCESSED_DIR
    / "trusted_change_zones_utm44n.gpkg"
)

QA_OUTPUT = (
    INTERIM_DIR
    / "spatial_change_zone_qa.csv"
)


# ============================================================
# SETTINGS
# ============================================================

CHANGE_MMU_M2 = 25.0


# ============================================================
# HELPERS
# ============================================================

def parse_bool(series):

    return (
        series
        .astype(str)
        .str.strip()
        .str.lower()
        .map(
            {
                "true": True,
                "false": False,
                "1": True,
                "0": False,
            }
        )
        .fillna(False)
        .astype(bool)
    )


def polygon_parts(geometry):
    """
    Extract individual polygon parts from any
    polygonal / geometry-collection result.
    """

    if geometry is None:
        return []

    if geometry.is_empty:
        return []

    geometry = make_valid(
        geometry
    )

    if isinstance(
        geometry,
        Polygon,
    ):
        return [geometry]

    if isinstance(
        geometry,
        MultiPolygon,
    ):
        return list(
            geometry.geoms
        )

    if isinstance(
        geometry,
        GeometryCollection,
    ):

        parts = []

        for sub_geometry in geometry.geoms:

            parts.extend(
                polygon_parts(
                    sub_geometry
                )
            )

        return parts

    return []


def polygonal_union(
    geometry,
):

    parts = polygon_parts(
        geometry
    )

    if not parts:
        return None

    return unary_union(
        parts
    )


def apply_mmu(
    geometry,
    minimum_area_m2,
):
    """
    Keep polygon components equal to or larger
    than the analysis MMU.

    Returns:
        filtered geometry
        raw area
        retained area
        removed area
        raw component count
        retained component count
    """

    parts = polygon_parts(
        geometry
    )

    raw_area = sum(
        part.area
        for part in parts
    )

    retained_parts = [
        part
        for part in parts
        if part.area >= minimum_area_m2
    ]

    retained_area = sum(
        part.area
        for part in retained_parts
    )

    removed_area = (
        raw_area
        - retained_area
    )

    if retained_parts:

        filtered_geometry = unary_union(
            retained_parts
        )

    else:

        filtered_geometry = None

    return (
        filtered_geometry,
        raw_area,
        retained_area,
        removed_area,
        len(parts),
        len(retained_parts),
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n========================================")
    print("TRUSTED SPATIAL LOSS / GAIN ANALYSIS")
    print("========================================")

    # --------------------------------------------------------
    # LOAD COMPARISON PAIRS
    # --------------------------------------------------------

    pairs = pd.read_csv(
        PAIR_INPUT
    )

    pairs[
        "manual_qa_required"
    ] = parse_bool(
        pairs[
            "manual_qa_required"
        ]
    )

    trusted_pairs = pairs[
        (
            pairs[
                "comparison_quality"
            ]
            .isin(
                [
                    "High",
                    "Moderate",
                ]
            )
        )
        &
        (
            ~pairs[
                "manual_qa_required"
            ]
        )
    ].copy()

    print(
        f"Trusted comparison pairs: "
        f"{len(trusted_pairs)}"
    )

    # --------------------------------------------------------
    # LOAD OBSERVATION GEOMETRIES
    # --------------------------------------------------------

    observations = gpd.read_file(
        OBSERVATION_INPUT
    )

    observations[
        "observation_date"
    ] = (
        pd.to_datetime(
            observations[
                "observation_date"
            ]
        )
        .dt.strftime(
            "%Y-%m-%d"
        )
    )

    print(
        f"Observation geometries loaded: "
        f"{len(observations)}"
    )

    print(
        f"CRS: "
        f"{observations.crs}"
    )

    # --------------------------------------------------------
    # BUILD GEOMETRY LOOKUP
    # --------------------------------------------------------

    geometry_lookup = {}

    for _, row in observations.iterrows():

        key = (
            row[
                "wetland_id"
            ],
            row[
                "observation_date"
            ],
        )

        geometry_lookup[
            key
        ] = make_valid(
            row.geometry
        )

    # --------------------------------------------------------
    # CALCULATE SPATIAL CHANGE
    # --------------------------------------------------------

    metric_rows = []
    zone_rows = []
    qa_rows = []

    missing_geometry_pairs = 0

    for _, pair in trusted_pairs.iterrows():

        wetland_id = pair[
            "wetland_id"
        ]

        start_date = str(
            pair[
                "start_date"
            ]
        )

        end_date = str(
            pair[
                "end_date"
            ]
        )

        start_key = (
            wetland_id,
            start_date,
        )

        end_key = (
            wetland_id,
            end_date,
        )

        if (
            start_key
            not in geometry_lookup
            or
            end_key
            not in geometry_lookup
        ):

            missing_geometry_pairs += 1

            continue

        start_geometry = geometry_lookup[
            start_key
        ]

        end_geometry = geometry_lookup[
            end_key
        ]

        # ----------------------------------------------------
        # RAW SPATIAL TRANSITIONS
        # ----------------------------------------------------

        stable_geometry = (
            start_geometry
            .intersection(
                end_geometry
            )
        )

        loss_geometry = (
            start_geometry
            .difference(
                end_geometry
            )
        )

        gain_geometry = (
            end_geometry
            .difference(
                start_geometry
            )
        )

        stable_geometry = polygonal_union(
            stable_geometry
        )

        # ----------------------------------------------------
        # APPLY 25 M2 MMU TO LOSS / GAIN ZONES
        # ----------------------------------------------------

        (
            loss_filtered,
            raw_loss_m2,
            analysis_loss_m2,
            removed_loss_m2,
            raw_loss_parts,
            retained_loss_parts,
        ) = apply_mmu(
            loss_geometry,
            CHANGE_MMU_M2,
        )

        (
            gain_filtered,
            raw_gain_m2,
            analysis_gain_m2,
            removed_gain_m2,
            raw_gain_parts,
            retained_gain_parts,
        ) = apply_mmu(
            gain_geometry,
            CHANGE_MMU_M2,
        )

        # ----------------------------------------------------
        # BASIC AREA VALUES
        # ----------------------------------------------------

        start_area_m2 = (
            start_geometry.area
        )

        end_area_m2 = (
            end_geometry.area
        )

        if (
            stable_geometry
            is not None
        ):

            stable_area_m2 = (
                stable_geometry.area
            )

        else:

            stable_area_m2 = 0.0

        raw_net_change_m2 = (
            raw_gain_m2
            - raw_loss_m2
        )

        analysis_net_change_m2 = (
            analysis_gain_m2
            - analysis_loss_m2
        )

        expected_net_change_m2 = (
            end_area_m2
            - start_area_m2
        )

        geometry_identity_error_m2 = abs(
            raw_net_change_m2
            - expected_net_change_m2
        )

        # ----------------------------------------------------
        # PERCENTAGES
        # ----------------------------------------------------

        if start_area_m2 > 0:

            stable_percent = (
                stable_area_m2
                / start_area_m2
                * 100.0
            )

            loss_percent = (
                analysis_loss_m2
                / start_area_m2
                * 100.0
            )

            gain_percent = (
                analysis_gain_m2
                / start_area_m2
                * 100.0
            )

            turnover_percent = (
                (
                    analysis_loss_m2
                    + analysis_gain_m2
                )
                / start_area_m2
                * 100.0
            )

        else:

            stable_percent = None
            loss_percent = None
            gain_percent = None
            turnover_percent = None

        # ----------------------------------------------------
        # SAVE PAIR METRICS
        # ----------------------------------------------------

        metric_rows.append(
            {
                "wetland_id":
                    wetland_id,

                "start_date":
                    start_date,

                "end_date":
                    end_date,

                "comparison_quality":
                    pair[
                        "comparison_quality"
                    ],

                "start_area_geom_ha":
                    start_area_m2
                    / 10000.0,

                "end_area_geom_ha":
                    end_area_m2
                    / 10000.0,

                "stable_area_ha":
                    stable_area_m2
                    / 10000.0,

                "raw_loss_area_ha":
                    raw_loss_m2
                    / 10000.0,

                "raw_gain_area_ha":
                    raw_gain_m2
                    / 10000.0,

                "analysis_loss_area_ha":
                    analysis_loss_m2
                    / 10000.0,

                "analysis_gain_area_ha":
                    analysis_gain_m2
                    / 10000.0,

                "raw_net_change_ha":
                    raw_net_change_m2
                    / 10000.0,

                "analysis_net_change_ha":
                    analysis_net_change_m2
                    / 10000.0,

                "stable_percent_of_start":
                    stable_percent,

                "loss_percent_of_start":
                    loss_percent,

                "gain_percent_of_start":
                    gain_percent,

                "turnover_percent_of_start":
                    turnover_percent,

                "geometry_identity_error_m2":
                    geometry_identity_error_m2,
            }
        )

        # ----------------------------------------------------
        # SAVE SPATIAL ZONES
        # ----------------------------------------------------

        base_attributes = {
            "wetland_id":
                wetland_id,

            "start_date":
                start_date,

            "end_date":
                end_date,

            "comparison_quality":
                pair[
                    "comparison_quality"
                ],
        }

        if (
            stable_geometry
            is not None
            and
            not stable_geometry.is_empty
        ):

            zone_rows.append(
                {
                    **base_attributes,

                    "change_type":
                        "Stable",

                    "area_ha":
                        stable_area_m2
                        / 10000.0,

                    "geometry":
                        stable_geometry,
                }
            )

        if (
            loss_filtered
            is not None
            and
            not loss_filtered.is_empty
        ):

            zone_rows.append(
                {
                    **base_attributes,

                    "change_type":
                        "Loss",

                    "area_ha":
                        analysis_loss_m2
                        / 10000.0,

                    "geometry":
                        loss_filtered,
                }
            )

        if (
            gain_filtered
            is not None
            and
            not gain_filtered.is_empty
        ):

            zone_rows.append(
                {
                    **base_attributes,

                    "change_type":
                        "Gain",

                    "area_ha":
                        analysis_gain_m2
                        / 10000.0,

                    "geometry":
                        gain_filtered,
                }
            )

        # ----------------------------------------------------
        # QA RECORD
        # ----------------------------------------------------

        qa_rows.append(
            {
                "wetland_id":
                    wetland_id,

                "start_date":
                    start_date,

                "end_date":
                    end_date,

                "raw_loss_components":
                    raw_loss_parts,

                "retained_loss_components":
                    retained_loss_parts,

                "raw_gain_components":
                    raw_gain_parts,

                "retained_gain_components":
                    retained_gain_parts,

                "removed_loss_area_m2":
                    removed_loss_m2,

                "removed_gain_area_m2":
                    removed_gain_m2,

                "geometry_identity_error_m2":
                    geometry_identity_error_m2,
            }
        )

    # ========================================================
    # SAVE METRICS
    # ========================================================

    metrics = pd.DataFrame(
        metric_rows
    )

    metrics.to_csv(
        METRICS_OUTPUT,
        index=False,
    )

    # ========================================================
    # SAVE QA
    # ========================================================

    qa = pd.DataFrame(
        qa_rows
    )

    qa.to_csv(
        QA_OUTPUT,
        index=False,
    )

    # ========================================================
    # SAVE SPATIAL CHANGE ZONES
    # ========================================================

    zones = gpd.GeoDataFrame(
        zone_rows,
        geometry="geometry",
        crs=observations.crs,
    )

    if ZONE_OUTPUT.exists():

        ZONE_OUTPUT.unlink()

    zones.to_file(
        ZONE_OUTPUT,
        layer="trusted_change_zones",
        driver="GPKG",
    )

    # ========================================================
    # TERMINAL SUMMARY
    # ========================================================

    print("\nSpatial change results:")

    print(
        f"Trusted pairs processed: "
        f"{len(metrics)}"
    )

    print(
        f"Pairs missing geometry: "
        f"{missing_geometry_pairs}"
    )

    print(
        f"Spatial zone records: "
        f"{len(zones)}"
    )

    print(
        "\nTotal trusted spatial change:"
    )

    print(
        f"Stable area summed across pairs: "
        f"{metrics['stable_area_ha'].sum():.4f} ha"
    )

    print(
        f"Raw loss: "
        f"{metrics['raw_loss_area_ha'].sum():.4f} ha"
    )

    print(
        f"Raw gain: "
        f"{metrics['raw_gain_area_ha'].sum():.4f} ha"
    )

    print(
        f"Analysis loss after MMU: "
        f"{metrics['analysis_loss_area_ha'].sum():.4f} ha"
    )

    print(
        f"Analysis gain after MMU: "
        f"{metrics['analysis_gain_area_ha'].sum():.4f} ha"
    )

    print(
        "\nMMU QA:"
    )

    print(
        f"Loss area removed by MMU: "
        f"{qa['removed_loss_area_m2'].sum():.2f} m²"
    )

    print(
        f"Gain area removed by MMU: "
        f"{qa['removed_gain_area_m2'].sum():.2f} m²"
    )

    print(
        f"Maximum geometry identity error: "
        f"{metrics['geometry_identity_error_m2'].max():.6f} m²"
    )

    print("\nOutputs:")

    print(
        METRICS_OUTPUT
    )

    print(
        ZONE_OUTPUT
    )

    print(
        QA_OUTPUT
    )

    print(
        "\nIMPORTANT:"
        "\nLoss and gain represent mapped boundary "
        "transitions between environmentally comparable "
        "Google Earth observations."
        "\nThey should not automatically be interpreted "
        "as human-caused degradation or restoration."
    )


if __name__ == "__main__":
    main()