
from pathlib import Path
from datetime import datetime, timezone
import json

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio

from rasterio.features import rasterize
from rasterio.enums import MergeAlg
from shapely.geometry import mapping


# ============================================================
# PART 3 - FUTURE WETLAND BOUNDARY PROJECTION
# SCRIPT 10 - GN-WISE WETLAND SCENARIO REPORTING
#
# Historical maps: 2015, 2020, 2025
# Exploratory scenario: 2030
#
# Pixel-centre assignment to 56 GN divisions.
# Each valid pixel is counted exactly once.
#
# No modification of source rasters or boundaries.
# ============================================================


ROOT = Path(__file__).resolve().parents[2]

MAP_DIR = ROOT / "data/processed/part3_future_projection"

METRIC_DIR = (
    ROOT / "outputs/metrics/part3_future_projection"
)

REPORT_DIR = (
    ROOT / "outputs/reports/part3_future_projection"
)

BOUNDARY_FILE = (
    ROOT / "data/raw/boundaries/Kaduwela TO Area.kml"
)

GN_LAYER = "GN Boundary Colombo District (#2)"

RASTER_FILES = {
    2015: MAP_DIR / "LULC_2015_6Class_CommonMask.tif",
    2020: MAP_DIR / "LULC_2020_6Class_CommonMask.tif",
    2025: MAP_DIR / "LULC_2025_6Class_CommonMask.tif",
    2030: MAP_DIR / "LULC_2030_Exploratory_CA_Markov.tif",
}

EXPECTED_GNS = 56
EXPECTED_VALID_PIXELS = 94258

WETLAND_CLASS = 1

# The source's "id" field is a source feature identifier,
# not necessarily an official administrative GN code.


def load_rasters():

    arrays = {}
    reference_grid = None

    transform = None
    crs = None

    for year, path in RASTER_FILES.items():

        if not path.is_file():
            raise FileNotFoundError(path)

        with rasterio.open(path) as src:

            grid = (
                src.crs,
                src.transform,
                src.width,
                src.height,
            )

            if reference_grid is None:
                reference_grid = grid
                transform = src.transform
                crs = src.crs

            elif grid != reference_grid:
                raise ValueError(
                    f"{year}: raster grid does not match."
                )

            if (
                str(src.crs) != "EPSG:32644"
                or src.count != 1
                or src.nodata != 0
            ):
                raise ValueError(
                    f"{year}: unexpected raster metadata."
                )

            arrays[year] = src.read(1)

    valid = arrays[2025] != 0

    if int(valid.sum()) != EXPECTED_VALID_PIXELS:
        raise ValueError(
            "Unexpected common valid pixel count."
        )

    for year, array in arrays.items():

        if not np.array_equal(
            valid,
            array != 0
        ):
            raise ValueError(
                f"{year}: common mask mismatch."
            )

        if not np.isin(
            array[valid],
            [1, 2, 3, 4, 5, 6]
        ).all():
            raise ValueError(
                f"{year}: invalid LULC class IDs."
            )

    pixel_area_m2 = abs(
        transform.a * transform.e
        - transform.b * transform.d
    )

    if not np.isclose(
        pixel_area_m2,
        900.0
    ):
        raise ValueError(
            "Expected 30 m Landsat pixels."
        )

    return (
        arrays,
        valid,
        transform,
        crs,
        pixel_area_m2 / 10000.0,
    )


def prepare_gn_assignment(
    valid,
    transform,
    crs,
):

    gn = gpd.read_file(
        BOUNDARY_FILE,
        layer=GN_LAYER,
    )

    if len(gn) != EXPECTED_GNS:
        raise ValueError(
            f"Expected 56 GN divisions, found {len(gn)}."
        )

    if gn.crs is None:
        raise ValueError(
            "GN boundary CRS is missing."
        )

    required = ["id", "Name", "geometry"]

    if not set(required).issubset(gn.columns):
        raise ValueError(
            "Required GN fields are missing."
        )

    gn = gn.to_crs(crs).copy()

    gn["Name"] = gn["Name"].astype(
        "string"
    ).str.strip()

    if (
        gn["Name"].isna().any()
        or (gn["Name"] == "").any()
        or not gn["Name"].is_unique
    ):
        raise ValueError(
            "GN names must be complete and unique."
        )

    if (
        gn["id"].isna().any()
        or not gn["id"].is_unique
    ):
        raise ValueError(
            "GN source identifiers must be unique."
        )

    # Stable ordering makes the raster zone assignments
    # reproducible, regardless of source feature order.
    gn = gn.sort_values(
        ["Name", "id"]
    ).reset_index(drop=True)

    gn["gn_zone"] = np.arange(
        1,
        len(gn) + 1,
        dtype=int
    )

    shapes_with_zones = []
    shapes_for_coverage = []

    repaired_geometries = []

    for _, row in gn.iterrows():

        geom = row.geometry

        if geom is None or geom.is_empty:
            raise ValueError(
                "Empty GN geometry."
            )

        if not geom.is_valid:
            geom = geom.buffer(0)

        if geom.is_empty or not geom.is_valid:
            raise ValueError(
                "Unusable GN geometry."
            )

        repaired_geometries.append(geom)

        shapes_with_zones.append(
            (mapping(geom), int(row["gn_zone"]))
        )

        shapes_for_coverage.append(
            (mapping(geom), 1)
        )

    gn = gn.set_geometry(
        gpd.GeoSeries(
            repaired_geometries,
            index=gn.index,
            crs=crs,
        )
    )

    # Independently verify coverage and overlaps.
    coverage = rasterize(
        shapes_for_coverage,
        out_shape=valid.shape,
        transform=transform,
        fill=0,
        dtype="uint16",
        all_touched=False,
        merge_alg=MergeAlg.add,
    )

    uncovered = int(
        (valid & (coverage == 0)).sum()
    )

    overlapping = int(
        (valid & (coverage > 1)).sum()
    )

    if uncovered != 0 or overlapping != 0:
        raise ValueError(
            "GN boundaries do not partition valid "
            "raster pixels uniquely. "
            f"Uncovered={uncovered}, "
            f"overlapping={overlapping}"
        )

    zone_raster = rasterize(
        shapes_with_zones,
        out_shape=valid.shape,
        transform=transform,
        fill=0,
        dtype="uint16",
        all_touched=False,
    )

    if not np.all(
        (zone_raster[valid] >= 1)
        & (zone_raster[valid] <= EXPECTED_GNS)
    ):
        raise ValueError(
            "Invalid GN zone assignments."
        )

    zone_pixel_counts = np.bincount(
        zone_raster[valid],
        minlength=EXPECTED_GNS + 1,
    )[1:]

    if int(zone_pixel_counts.sum()) != int(
        valid.sum()
    ):
        raise ValueError(
            "GN pixel accounting failed."
        )

    return gn, zone_raster, zone_pixel_counts


def count_by_zone(mask, zones):
    """
    Count true pixels inside each GN zone.
    Zone 0 is outside the GN reporting area.
    """

    counts = np.bincount(
        zones[mask],
        minlength=EXPECTED_GNS + 1,
    )[1:]

    return counts.astype(np.int64)


def main():

    print("=" * 70)
    print("PART 3 - FUTURE WETLAND BOUNDARY PROJECTION")
    print("SCRIPT 10 - GN-WISE WETLAND SCENARIO REPORTING")
    print("=" * 70)

    (
        maps,
        valid,
        transform,
        crs,
        pixel_area_ha,
    ) = load_rasters()

    (
        gn,
        zone_raster,
        zone_pixel_counts,
    ) = prepare_gn_assignment(
        valid,
        transform,
        crs,
    )

    print("\nINPUT VALIDATION: PASS")
    print("GN divisions:", len(gn))
    print("Common valid pixels:", int(valid.sum()))
    print("Unassigned valid pixels: 0")
    print("Overlapping GN pixel assignments: 0")
    print("Pixel area:", pixel_area_ha, "ha")

    # --------------------------------------------------------
    # 1. HISTORICAL AND SCENARIO WETLAND MASKS
    # --------------------------------------------------------

    wetland_masks = {
        year: (
            valid
            & (maps[year] == WETLAND_CLASS)
        )
        for year in RASTER_FILES
    }

    wetland_2025 = wetland_masks[2025]
    wetland_2030 = wetland_masks[2030]

    persistence = (
        wetland_2025 & wetland_2030
    )

    loss = (
        wetland_2025 & ~wetland_2030
    )

    gain = (
        valid
        & ~wetland_2025
        & wetland_2030
    )

    # --------------------------------------------------------
    # 2. COUNT PIXELS PER GN
    # --------------------------------------------------------

    results = pd.DataFrame({
        "gn_zone": gn["gn_zone"].to_numpy(),
        "gn_source_id": gn["id"].astype(str).to_numpy(),
        "gn_name": gn["Name"].astype(str).to_numpy(),
        "analysis_pixels": zone_pixel_counts,
    })

    results["analysis_area_ha"] = (
        results["analysis_pixels"] * pixel_area_ha
    )

    for year in RASTER_FILES:

        pixel_column = f"wetland_{year}_pixels"
        area_column = f"wetland_{year}_ha"

        results[pixel_column] = count_by_zone(
            wetland_masks[year],
            zone_raster,
        )

        results[area_column] = (
            results[pixel_column] * pixel_area_ha
        )

    change_masks = {
        "persistence": persistence,
        "scenario_loss": loss,
        "scenario_gain": gain,
    }

    for name, mask in change_masks.items():

        results[f"{name}_pixels"] = count_by_zone(
            mask,
            zone_raster,
        )

        results[f"{name}_ha"] = (
            results[f"{name}_pixels"]
            * pixel_area_ha
        )

    results["scenario_net_change_pixels"] = (
        results["scenario_gain_pixels"]
        - results["scenario_loss_pixels"]
    )

    results["scenario_net_change_ha"] = (
        results["scenario_net_change_pixels"]
        * pixel_area_ha
    )

    results["wetland_2025_pct_of_gn"] = (
        100
        * results["wetland_2025_pixels"]
        / results["analysis_pixels"]
    )

    results["wetland_2030_pct_of_gn"] = (
        100
        * results["wetland_2030_pixels"]
        / results["analysis_pixels"]
    )

    results["scenario_net_change_pct_of_gn"] = (
        100
        * results["scenario_net_change_pixels"]
        / results["analysis_pixels"]
    )

    # --------------------------------------------------------
    # 3. VERIFY EACH GN'S WETLAND ACCOUNTING
    # --------------------------------------------------------

    if not (
        results["wetland_2025_pixels"]
        == (
            results["persistence_pixels"]
            + results["scenario_loss_pixels"]
        )
    ).all():
        raise ValueError(
            "GN-level 2025 wetland accounting failed."
        )

    if not (
        results["wetland_2030_pixels"]
        == (
            results["persistence_pixels"]
            + results["scenario_gain_pixels"]
        )
    ).all():
        raise ValueError(
            "GN-level 2030 wetland accounting failed."
        )

    if not (
        results["scenario_net_change_pixels"]
        == (
            results["wetland_2030_pixels"]
            - results["wetland_2025_pixels"]
        )
    ).all():
        raise ValueError(
            "GN-level net-change accounting failed."
        )

    # --------------------------------------------------------
    # 4. VERIFY GLOBAL TOTALS
    # --------------------------------------------------------

    global_checks = {
        "analysis_pixels": int(valid.sum()),
        "wetland_2015_pixels": int(
            wetland_masks[2015].sum()
        ),
        "wetland_2020_pixels": int(
            wetland_masks[2020].sum()
        ),
        "wetland_2025_pixels": int(
            wetland_masks[2025].sum()
        ),
        "wetland_2030_pixels": int(
            wetland_masks[2030].sum()
        ),
        "persistence_pixels": int(
            persistence.sum()
        ),
        "scenario_loss_pixels": int(
            loss.sum()
        ),
        "scenario_gain_pixels": int(
            gain.sum()
        ),
    }

    for column, expected in global_checks.items():

        actual = int(
            results[column].sum()
        )

        if actual != expected:
            raise ValueError(
                f"Global GN accounting mismatch for {column}: "
                f"{actual} vs {expected}"
            )

    # Cross-check against outputs already reviewed
    # in Scripts 05, 08 and 09.
    expected_previous_results = {
        "wetland_2015_pixels": 19523,
        "wetland_2020_pixels": 12217,
        "wetland_2025_pixels": 15401,
        "wetland_2030_pixels": 15883,
        "persistence_pixels": 5427,
        "scenario_loss_pixels": 9974,
        "scenario_gain_pixels": 10456,
    }

    for column, expected in expected_previous_results.items():

        if global_checks[column] != expected:
            raise ValueError(
                f"Results differ from previous scripts: "
                f"{column}"
            )

    print("\nGN AND GLOBAL ACCOUNTING: PASS")

    # --------------------------------------------------------
    # 5. SAVE CSV AND GIS OUTPUTS
    # --------------------------------------------------------

    METRIC_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    MAP_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    csv_path = (
        METRIC_DIR
        / "script10_gn_wise_wetland_scenario_summary.csv"
    )

    gpkg_utm_path = (
        MAP_DIR
        / "GN_Wise_Wetland_Scenario_UTM44N.gpkg"
    )

    gpkg_wgs_path = (
        MAP_DIR
        / "GN_Wise_Wetland_Scenario_WGS84.gpkg"
    )

    metadata_path = (
        REPORT_DIR
        / "script10_gn_wise_scenario_metadata.json"
    )

    # Keep unrounded values during validation.
    # Round reported areas only for presentation.
    area_columns = [
        column
        for column in results.columns
        if column.endswith("_ha")
    ]

    percent_columns = [
        column
        for column in results.columns
        if "_pct_" in column
    ]

    export = results.copy()

    export[area_columns] = (
        export[area_columns].round(4)
    )

    export[percent_columns] = (
        export[percent_columns].round(3)
    )

    export.to_csv(
        csv_path,
        index=False,
    )

    spatial = gn[
        ["gn_zone", "geometry"]
    ].merge(
        export,
        on="gn_zone",
        how="left",
        validate="one_to_one",
    )

    if (
        len(spatial) != EXPECTED_GNS
        or spatial["gn_name"].isna().any()
    ):
        raise ValueError(
            "GN spatial attribute join failed."
        )

    if gpkg_utm_path.exists():
        gpkg_utm_path.unlink()

    spatial.to_file(
        gpkg_utm_path,
        layer="gn_wetland_scenario",
        driver="GPKG",
        index=False,
    )

    spatial_wgs = spatial.to_crs(
        "EPSG:4326"
    )

    if gpkg_wgs_path.exists():
        gpkg_wgs_path.unlink()

    spatial_wgs.to_file(
        gpkg_wgs_path,
        layer="gn_wetland_scenario",
        driver="GPKG",
        index=False,
    )

    # Check saved GeoPackages can be opened.
    for path in [
        gpkg_utm_path,
        gpkg_wgs_path,
    ]:

        saved = gpd.read_file(
            path,
            layer="gn_wetland_scenario",
        )

        if (
            len(saved) != EXPECTED_GNS
            or saved["gn_name"].isna().any()
            or int(saved["analysis_pixels"].sum())
            != EXPECTED_VALID_PIXELS
        ):
            raise ValueError(
                f"GeoPackage read-back failed: {path}"
            )

    # --------------------------------------------------------
    # 6. SAVE RESEARCH METADATA
    # --------------------------------------------------------

    metadata = {
        "component": (
            "Part 3 - Future Wetland Boundary Projection"
        ),
        "stage": "GN-wise wetland scenario reporting",
        "generated_utc":
            datetime.now(timezone.utc).isoformat(),
        "status": "REPORTING_COMPLETE_SCENARIO_UNVALIDATED",
        "gn_count": EXPECTED_GNS,
        "study_area_source": str(BOUNDARY_FILE),
        "gn_source_layer": GN_LAYER,
        "gn_name_field": "Name",
        "gn_identifier_field": "id",
        "gn_identifier_note": (
            "Source feature ID is not assumed to be "
            "an official GN administrative code."
        ),
        "reporting_method": (
            "Unique pixel-centre assignment to GN polygons"
        ),
        "reporting_crs": "EPSG:32644",
        "dashboard_crs": "EPSG:4326",
        "pixel_area_ha": pixel_area_ha,
        "valid_pixel_count": int(valid.sum()),
        "gn_overlap_pixels": 0,
        "unassigned_valid_pixels": 0,
        "global_totals_pixels": global_checks,
        "all_small_wetland_patches_included": True,
        "outputs": {
            "csv": str(csv_path),
            "gpkg_utm": str(gpkg_utm_path),
            "gpkg_wgs84": str(gpkg_wgs_path),
        },
        "validation": {
            "56_unique_gn_names": True,
            "all_pixels_uniquely_assigned": True,
            "individual_gn_accounting": True,
            "study_area_total_accounting": True,
            "previous_script_totals_match": True,
            "gis_file_readback": True,
        },
        "limitations": [
            "2015, 2020 and 2025 areas originate from "
            "model-classified LULC maps.",
            "2030 areas are an exploratory CA-Markov scenario.",
            "The prior 2025 hindcast underperformed persistence.",
            "GN-level gains/losses are scenario outputs, "
            "not confirmed future environmental changes.",
            "Pixel-centre assignment gives raster-equivalent "
            "areas, not precise polygon intersection areas.",
            "Each GN area includes only pixels within "
            "the three-year common valid raster mask.",
            "No legal conservation or protection status "
            "is inferred from these outputs.",
        ],
    }

    metadata_path.write_text(
        json.dumps(
            metadata,
            indent=2,
        ),
        encoding="utf-8",
    )

    # --------------------------------------------------------
    # 7. TERMINAL SUMMARY
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("STUDY-AREA WETLAND TOTALS")
    print("=" * 70)

    for year in RASTER_FILES:

        total = float(
            results[f"wetland_{year}_ha"].sum()
        )

        print(
            f"{year}: {total:.2f} ha"
        )

    for name in [
        "persistence",
        "scenario_loss",
        "scenario_gain",
    ]:

        print(
            f"{name}: "
            f"{results[f'{name}_ha'].sum():.2f} ha"
        )

    print(
        "Scenario net change:",
        f"{results['scenario_net_change_ha'].sum():+.2f} ha"
    )

    print("\nFIVE LARGEST GN SCENARIO NET LOSSES")

    print(
        export.nsmallest(
            5,
            "scenario_net_change_ha"
        )[
            [
                "gn_name",
                "wetland_2025_ha",
                "wetland_2030_ha",
                "scenario_net_change_ha",
            ]
        ].to_string(index=False)
    )

    print("\nFIVE LARGEST GN SCENARIO NET GAINS")

    print(
        export.nlargest(
            5,
            "scenario_net_change_ha"
        )[
            [
                "gn_name",
                "wetland_2025_ha",
                "wetland_2030_ha",
                "scenario_net_change_ha",
            ]
        ].to_string(index=False)
    )

    print("\nSAVED OUTPUTS")
    print(csv_path)
    print(gpkg_utm_path)
    print(gpkg_wgs_path)
    print(metadata_path)

    print("\nSCRIPT 10: COMPLETED")
    print("GN DIVISIONS REPORTED: 56")
    print("PIXEL ACCOUNTING: PASS")
    print("SCENARIO VALIDATION: NOT ESTABLISHED")


if __name__ == "__main__":
    main()
