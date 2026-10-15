
from pathlib import Path
from datetime import datetime, timezone
import json

import geopandas as gpd
import numpy as np
import pandas as pd


# ============================================================
# PART 3 - SCRIPT 11
# FRONTEND-READY GIS HANDOVER LAYERS
#
# Converts model-derived wetland polygons:
# EPSG:32644 -> EPSG:4326
#
# Original GeoPackages are never modified.
# All patches, including small patches, are retained.
# Area attributes remain calculated in UTM metres.
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = (
    ROOT / "data/processed/part3_future_projection"
)

METRIC_DIR = (
    ROOT / "outputs/metrics/part3_future_projection"
)

REPORT_DIR = (
    ROOT / "outputs/reports/part3_future_projection"
)

SUMMARY_FILE = (
    METRIC_DIR / "script09_wetland_patch_summary.csv"
)

SOURCE_LAYERS = {
    "wetland_2025": "Wetland_2025_Estimated_Patches.gpkg",
    "wetland_2030": "Wetland_2030_Exploratory_Patches.gpkg",
    "persistence": "Wetland_2025_2030_Persistence.gpkg",
    "scenario_loss": "Wetland_2025_2030_Scenario_Loss.gpkg",
    "scenario_gain": "Wetland_2025_2030_Scenario_Gain.gpkg",
}

EXPECTED_SOURCE_CRS = "EPSG:32644"
OUTPUT_CRS = "EPSG:4326"

GN_WGS_FILE = (
    DATA_DIR / "GN_Wise_Wetland_Scenario_WGS84.gpkg"
)


def validate_source(gdf, layer_name, expected):

    if gdf.crs is None:
        raise ValueError(
            f"{layer_name}: source CRS missing"
        )

    if gdf.crs.to_epsg() != 32644:
        raise ValueError(
            f"{layer_name}: unexpected source CRS"
        )

    required = [
        "patch_id",
        "patch_type",
        "pixel_count",
        "area_ha",
        "display_ge5",
        "source_type",
        "geometry",
    ]

    missing = [
        col for col in required
        if col not in gdf.columns
    ]

    if missing:
        raise ValueError(
            f"{layer_name}: missing columns {missing}"
        )

    if len(gdf) != int(expected["patch_count"]):
        raise ValueError(
            f"{layer_name}: source feature count mismatch"
        )

    if int(gdf["pixel_count"].sum()) != int(
        expected["all_pixels"]
    ):
        raise ValueError(
            f"{layer_name}: pixel accounting mismatch"
        )

    if not np.isclose(
        gdf["area_ha"].sum(),
        float(expected["all_area_ha"]),
        atol=0.001,
    ):
        raise ValueError(
            f"{layer_name}: area accounting mismatch"
        )

    if gdf["patch_id"].duplicated().any():
        raise ValueError(
            f"{layer_name}: duplicate patch identifiers"
        )

    if not gdf.geometry.notna().all():
        raise ValueError(
            f"{layer_name}: missing geometry"
        )

    if not (
        gdf.geometry.is_valid
        & ~gdf.geometry.is_empty
    ).all():
        raise ValueError(
            f"{layer_name}: invalid source geometry"
        )

    if not gdf["display_ge5"].isin([0, 1]).all():
        raise ValueError(
            f"{layer_name}: invalid display flag"
        )


def main():

    print("=" * 70)
    print("PART 3 - FRONTEND GIS HANDOVER")
    print("SCRIPT 11 - WGS84 WETLAND SCREENING LAYERS")
    print("=" * 70)

    if not SUMMARY_FILE.is_file():
        raise FileNotFoundError(SUMMARY_FILE)

    if not GN_WGS_FILE.is_file():
        raise FileNotFoundError(GN_WGS_FILE)

    summary = pd.read_csv(SUMMARY_FILE)

    if set(summary["layer"]) != set(SOURCE_LAYERS):
        raise ValueError(
            "Script 09 patch summary does not match "
            "the expected five layers."
        )

    summary = summary.set_index("layer")

    # Verify the GN reporting layer already produced
    # by Script 10 is suitable for a WGS84 web map.
    gn = gpd.read_file(
        GN_WGS_FILE,
        layer="gn_wetland_scenario",
    )

    if (
        len(gn) != 56
        or gn.crs is None
        or gn.crs.to_epsg() != 4326
        or int(gn["analysis_pixels"].sum()) != 94258
    ):
        raise ValueError(
            "GN WGS84 handover layer is inconsistent."
        )

    print("\nGN WGS84 layer: PASS")
    print("GN divisions:", len(gn))

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    records = []
    outputs = {}

    for layer_name, source_filename in SOURCE_LAYERS.items():

        print("\n" + "-" * 65)
        print("Processing:", layer_name)

        source_path = DATA_DIR / source_filename

        if not source_path.is_file():
            raise FileNotFoundError(source_path)

        source = gpd.read_file(
            source_path,
            layer=layer_name,
        )

        validate_source(
            source,
            layer_name,
            summary.loc[layer_name],
        )

        output_path = (
            DATA_DIR
            / f"Part3_{layer_name}_WGS84.gpkg"
        )

        # Reproject geometries only.
        # Retain UTM-derived area_ha attributes.
        converted = source.to_crs(OUTPUT_CRS)

        if output_path.exists():
            output_path.unlink()

        converted.to_file(
            output_path,
            layer=layer_name,
            driver="GPKG",
            index=False,
        )

        # Read back the output and independently
        # check important numerical attributes.
        saved = gpd.read_file(
            output_path,
            layer=layer_name,
        )

        if (
            saved.crs is None
            or saved.crs.to_epsg() != 4326
            or len(saved) != len(source)
        ):
            raise ValueError(
                f"{layer_name}: WGS84 output check failed"
            )

        if set(saved["patch_id"]) != set(
            source["patch_id"]
        ):
            raise ValueError(
                f"{layer_name}: patch identities changed"
            )

        if int(saved["pixel_count"].sum()) != int(
            source["pixel_count"].sum()
        ):
            raise ValueError(
                f"{layer_name}: pixel total changed"
            )

        if not np.isclose(
            saved["area_ha"].sum(),
            source["area_ha"].sum(),
            atol=0.001,
        ):
            raise ValueError(
                f"{layer_name}: area attributes changed"
            )

        if not (
            saved.geometry.is_valid
            & ~saved.geometry.is_empty
        ).all():
            raise ValueError(
                f"{layer_name}: invalid exported geometry"
            )

        records.append({
            "layer": layer_name,
            "features": int(len(saved)),
            "total_pixels": int(
                saved["pixel_count"].sum()
            ),
            "area_ha": round(
                float(saved["area_ha"].sum()),
                4,
            ),
            "crs": OUTPUT_CRS,
            "file": output_path.name,
        })

        outputs[layer_name] = str(output_path)

        print("Features:", len(saved))
        print(
            "Area:",
            f"{saved['area_ha'].sum():.2f} ha"
        )
        print("Output CRS:", saved.crs)
        print("Geometry and accounting checks: PASS")
        print("Saved:", output_path.name)

    manifest = {
        "component": (
            "Part 3 - Future Wetland Boundary Projection"
        ),
        "stage": "GIS handover layer preparation",
        "created_utc":
            datetime.now(timezone.utc).isoformat(),
        "status": "GIS_EXPORT_COMPLETED",
        "source_crs": EXPECTED_SOURCE_CRS,
        "output_crs": OUTPUT_CRS,
        "area_units": "hectares",
        "area_measurement_crs": EXPECTED_SOURCE_CRS,
        "gn_count": 56,
        "gn_layer": str(GN_WGS_FILE),
        "polygon_layers": outputs,
        "layer_summaries": records,
        "small_patches_removed": False,
        "interpretation": (
            "Model-derived historical and exploratory "
            "future wetland screening layers."
        ),
        "limitations": [
            "2030 is an exploratory CA-Markov scenario.",
            "Hindcast did not outperform persistence.",
            "Polygons are not legally established or "
            "field-surveyed wetland boundaries.",
            "Area attributes originate from the 30 m "
            "UTM raster grid, not degree-based area.",
            "All small polygons remain included.",
            "Map users must not treat scenario gain "
            "or loss as confirmed environmental change.",
        ],
    }

    manifest_path = (
        REPORT_DIR
        / "script11_gis_handover_manifest.json"
    )

    summary_path = (
        METRIC_DIR
        / "script11_gis_handover_summary.csv"
    )

    manifest_path.write_text(
        json.dumps(
            manifest,
            indent=2,
        ),
        encoding="utf-8",
    )

    pd.DataFrame(records).to_csv(
        summary_path,
        index=False,
    )

    print("\n" + "=" * 70)
    print("SCRIPT 11 FINAL SUMMARY")
    print("=" * 70)

    print(
        pd.DataFrame(records)[
            ["layer", "features", "area_ha"]
        ].to_string(index=False)
    )

    print("\nWGS84 wetland layers generated:", len(records))
    print("GN WGS84 layer: verified")
    print("Original projected files: unchanged")

    print("\nSaved manifest:", manifest_path)
    print("Saved summary:", summary_path)

    print("\nSCRIPT 11: COMPLETED")
    print("GIS HANDOVER PREPARATION: PASSED")
    print("FINAL BACKEND REVIEW: STILL REQUIRED")


if __name__ == "__main__":
    main()
