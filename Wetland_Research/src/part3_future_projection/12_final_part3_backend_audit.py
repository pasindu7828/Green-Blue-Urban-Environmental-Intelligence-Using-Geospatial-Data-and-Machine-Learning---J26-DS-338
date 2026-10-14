
from pathlib import Path
from datetime import datetime, timezone
import json

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio

from rasterio.features import rasterize
from rasterio.enums import MergeAlg


ROOT = Path(__file__).resolve().parents[2]

DATA = ROOT / "data/processed/part3_future_projection"
METRICS = ROOT / "outputs/metrics/part3_future_projection"
REPORTS = ROOT / "outputs/reports/part3_future_projection"

BOUNDARY = ROOT / "data/raw/boundaries/Kaduwela TO Area.kml"
GN_LAYER = "GN Boundary Colombo District (#2)"

YEARS = {
    2015: "LULC_2015_6Class_CommonMask.tif",
    2020: "LULC_2020_6Class_CommonMask.tif",
    2025: "LULC_2025_6Class_CommonMask.tif",
    2030: "LULC_2030_Exploratory_CA_Markov.tif",
}

LAYERS = {
    "wetland_2025": (
        "Part3_wetland_2025_WGS84.gpkg",
        "Wetland_2025_Estimated_Patches.gpkg",
    ),
    "wetland_2030": (
        "Part3_wetland_2030_WGS84.gpkg",
        "Wetland_2030_Exploratory_Patches.gpkg",
    ),
    "persistence": (
        "Part3_persistence_WGS84.gpkg",
        "Wetland_2025_2030_Persistence.gpkg",
    ),
    "scenario_loss": (
        "Part3_scenario_loss_WGS84.gpkg",
        "Wetland_2025_2030_Scenario_Loss.gpkg",
    ),
    "scenario_gain": (
        "Part3_scenario_gain_WGS84.gpkg",
        "Wetland_2025_2030_Scenario_Gain.gpkg",
    ),
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    print("=" * 68)
    print("PART 3 - FINAL BACKEND CONSISTENCY AUDIT")
    print("=" * 68)

    # 1. Check original classified and scenario rasters.
    maps = {}
    grid = None

    for year, filename in YEARS.items():
        with rasterio.open(DATA / filename) as src:
            metadata = (
                src.crs, src.transform,
                src.width, src.height,
            )
            if grid is None:
                grid = metadata
                transform = src.transform
                crs = src.crs
                raster_shape = (src.height, src.width)
            else:
                require(
                    metadata == grid,
                    f"{year}: raster grids differ"
                )

            require(
                src.count == 1 and src.nodata == 0,
                f"{year}: raster encoding mismatch"
            )
            maps[year] = src.read(1)

    require(str(crs) == "EPSG:32644", "Unexpected CRS")

    valid = maps[2015] != 0
    require(int(valid.sum()) == 94258, "Mask size mismatch")

    for year, array in maps.items():
        require(
            np.array_equal(array != 0, valid),
            f"{year}: common mask mismatch"
        )
        require(
            np.isin(array[valid], range(1, 7)).all(),
            f"{year}: invalid class values"
        )

    pixel_ha = abs(
        transform.a * transform.e
        - transform.b * transform.d
    ) / 10000.0

    require(np.isclose(pixel_ha, 0.09), "Pixel area mismatch")

    print("\nRASTER GRID AND COMMON MASK: PASS")

    # 2. Reconstruct analytical wetland masks.
    w25 = valid & (maps[2025] == 1)
    w30 = valid & (maps[2030] == 1)

    masks = {
        "wetland_2025": w25,
        "wetland_2030": w30,
        "persistence": w25 & w30,
        "scenario_loss": w25 & ~w30,
        "scenario_gain": valid & ~w25 & w30,
    }

    require(
        np.array_equal(
            masks["persistence"] | masks["scenario_loss"],
            w25
        ),
        "2025 wetland change accounting failed"
    )
    require(
        np.array_equal(
            masks["persistence"] | masks["scenario_gain"],
            w30
        ),
        "2030 wetland change accounting failed"
    )

    print("WETLAND PERSISTENCE/LOSS/GAIN ACCOUNTING: PASS")

    # 3. Verify each GIS polygon layer against raster pixels.
    layer_results = {}

    print("\nCHECKING EXPORTED GIS POLYGONS")

    for name, (wgs_file, utm_file) in LAYERS.items():
        wgs = gpd.read_file(
            DATA / wgs_file,
            layer=name
        )
        utm = gpd.read_file(
            DATA / utm_file,
            layer=name
        )

        require(
            wgs.crs.to_epsg() == 4326,
            f"{name}: WGS84 CRS mismatch"
        )
        require(
            utm.crs.to_epsg() == 32644,
            f"{name}: projected CRS mismatch"
        )
        require(
            len(wgs) == len(utm),
            f"{name}: polygon count mismatch"
        )
        require(
            set(wgs["patch_id"]) == set(utm["patch_id"]),
            f"{name}: patch ID mismatch"
        )
        require(
            int(wgs["pixel_count"].sum())
            == int(masks[name].sum()),
            f"{name}: pixel-count mismatch"
        )
        require(
            np.isclose(
                wgs["area_ha"].sum(),
                masks[name].sum() * pixel_ha,
                atol=0.001
            ),
            f"{name}: area mismatch"
        )

        require(
            wgs.geometry.is_valid.all()
            and not wgs.geometry.is_empty.any(),
            f"{name}: invalid WGS84 geometry"
        )

        # Reproject WGS84 polygons to the source raster
        # grid and check their exact pixel-centre coverage.
        back_projected = wgs.to_crs(crs)

        reconstructed = rasterize(
            (
                (geom, 1)
                for geom in back_projected.geometry
            ),
            out_shape=raster_shape,
            transform=transform,
            fill=0,
            dtype="uint8",
            all_touched=False
        ).astype(bool)

        mismatched_pixels = int(
            np.count_nonzero(
                reconstructed != masks[name]
            )
        )

        require(
            mismatched_pixels == 0,
            f"{name}: {mismatched_pixels} raster pixels "
            "differ from GIS polygons"
        )

        layer_results[name] = {
            "polygons": len(wgs),
            "pixels": int(masks[name].sum()),
            "area_ha": round(
                float(masks[name].sum() * pixel_ha), 2
            ),
            "raster_vector_mismatched_pixels": 0,
        }

        print(
            f"  {name:18s}: "
            f"{len(wgs):5d} polygons | "
            f"{layer_results[name]['area_ha']:8.2f} ha | "
            "PASS"
        )

    # 4. Rebuild GN pixel assignments and check reporting.
    gn = gpd.read_file(
        BOUNDARY,
        layer=GN_LAYER
    ).to_crs(crs)

    require(len(gn) == 56, "GN count mismatch")
    require(gn["Name"].is_unique, "Duplicate GN names")

    geometries = []

    for geom in gn.geometry:
        if not geom.is_valid:
            geom = geom.buffer(0)
        require(not geom.is_empty, "Empty GN geometry")
        geometries.append(geom)

    coverage = rasterize(
        ((g, 1) for g in geometries),
        out_shape=raster_shape,
        transform=transform,
        fill=0,
        dtype="uint16",
        merge_alg=MergeAlg.add,
        all_touched=False,
    )

    require(
        np.all(coverage[valid] == 1),
        "GN coverage overlaps or gaps found"
    )

    gn_map = rasterize(
        (
            (geom, i + 1)
            for i, geom in enumerate(geometries)
        ),
        out_shape=raster_shape,
        transform=transform,
        fill=0,
        dtype="uint16",
        all_touched=False,
    )

    summary = pd.read_csv(
        METRICS / "script10_gn_wise_wetland_scenario_summary.csv"
    )

    require(len(summary) == 56, "GN summary row count mismatch")

    for i, (_, row) in enumerate(gn.iterrows()):
        name = str(row["Name"]).strip()
        matching = summary[summary["gn_name"] == name]

        require(
            len(matching) == 1,
            f"{name}: missing/duplicate GN summary"
        )

        record = matching.iloc[0]
        zone = valid & (gn_map == i + 1)

        expected = {
            "analysis_pixels": int(zone.sum()),
            "wetland_2015_pixels": int(
                (zone & (maps[2015] == 1)).sum()
            ),
            "wetland_2020_pixels": int(
                (zone & (maps[2020] == 1)).sum()
            ),
            "wetland_2025_pixels": int(
                (zone & w25).sum()
            ),
            "wetland_2030_pixels": int(
                (zone & w30).sum()
            ),
        }

        for layer in [
            "persistence", "scenario_loss", "scenario_gain"
        ]:
            expected[f"{layer}_pixels"] = int(
                (zone & masks[layer]).sum()
            )

        for column, pixel_count in expected.items():
            require(
                int(record[column]) == pixel_count,
                f"{name}: {column} mismatch"
            )

        net_change = (
            expected["scenario_gain_pixels"]
            - expected["scenario_loss_pixels"]
        )

        require(
            int(record["scenario_net_change_pixels"])
            == net_change,
            f"{name}: net scenario change mismatch"
        )

    require(
        int(summary["analysis_pixels"].sum()) == 94258,
        "GN summary global pixel accounting mismatch"
    )

    gn_wgs = gpd.read_file(
        DATA / "GN_Wise_Wetland_Scenario_WGS84.gpkg",
        layer="gn_wetland_scenario"
    )

    require(
        len(gn_wgs) == 56
        and gn_wgs.crs.to_epsg() == 4326,
        "WGS84 GN GeoPackage invalid"
    )

    print("\nGN-WISE RASTER AND CSV ACCOUNTING: PASS")
    print("GN WGS84 GIS LAYER: PASS")

    # 5. Confirm the hindcast comparison evidence exists.
    hindcast = pd.read_csv(
        METRICS / "script07_hindcast_model_comparison.csv"
    )

    methods = hindcast.set_index("method")

    require(
        {
            "Persistence",
            "Markov_RandomAllocation",
            "Markov_NeighbourhoodCA",
        }.issubset(methods.index),
        "Missing hindcast comparison methods"
    )

    persistence_accuracy = float(
        methods.loc["Persistence", "accuracy"]
    )
    ca_accuracy = float(
        methods.loc["Markov_NeighbourhoodCA", "accuracy"]
    )

    ca_outperformed_persistence = (
        ca_accuracy > persistence_accuracy
    )

    print("\nHINDCAST EVIDENCE")
    print(
        f"Persistence accuracy: {persistence_accuracy:.4f}"
    )
    print(f"CA-Markov accuracy: {ca_accuracy:.4f}")
    print(
        "CA outperformed persistence:",
        ca_outperformed_persistence
    )

    # 6. Save the final consistency report.
    REPORTS.mkdir(parents=True, exist_ok=True)

    output = REPORTS / "script12_final_backend_audit.json"

    report = {
        "component": "Part 3 - Future Wetland Boundary Projection",
        "generated_utc":
            datetime.now(timezone.utc).isoformat(),
        "technical_consistency": "PASS",
        "gn_divisions": 56,
        "common_valid_pixels": int(valid.sum()),
        "pixel_area_ha": pixel_ha,
        "gis_layers": layer_results,
        "gn_reporting_consistency": "PASS",
        "raster_vector_pixel_agreement": "PASS",
        "hindcast": {
            "persistence_accuracy": persistence_accuracy,
            "ca_markov_accuracy": ca_accuracy,
            "ca_outperformed_persistence":
                ca_outperformed_persistence,
        },
        "scenario_status": "EXPLORATORY_UNVALIDATED",
        "important_limitations": [
            "Technical consistency is not predictive validation.",
            "Historical LULC classifications have errors.",
            "The 2025 CA-Markov hindcast underperformed "
            "the persistence baseline.",
            "2030 wetland boundaries and changes are "
            "scenario-derived screening products.",
            "No confirmed 2030 wetland conditions exist "
            "for validation.",
            "These layers must not be treated as official "
            "legal or conservation boundaries.",
        ],
    }

    output.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8"
    )

    print("\n" + "=" * 68)
    print("SCRIPT 12 - FINAL RESULT")
    print("=" * 68)
    print("Raster consistency: PASS")
    print("GIS polygon consistency: PASS")
    print("GN reporting consistency: PASS")
    print("Hindcast evidence preserved: PASS")
    print("2030 forecast validity: NOT ESTABLISHED")
    print("\nSaved:", output)
    print("\nFINAL BACKEND AUDIT: COMPLETED")


if __name__ == "__main__":
    main()
