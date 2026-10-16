
from pathlib import Path
from datetime import datetime, timezone
import json

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio

from rasterio.features import shapes
from scipy.ndimage import label
from shapely.geometry import shape
from shapely.ops import unary_union

try:
    from shapely import make_valid
except ImportError:
    from shapely.validation import make_valid


# ============================================================
# PART 3 - FUTURE WETLAND BOUNDARY PROJECTION
# SCRIPT 09 - WETLAND POLYGON AND CHANGE SCREENING
#
# Inputs:
# - Classified 2025 LULC raster
# - Exploratory 2030 CA-Markov LULC raster
#
# Outputs:
# - Complete 2025 wetland polygons
# - Complete 2030 scenario wetland polygons
# - Scenario persistence, loss, and gain polygons
# - Patch summary and area validation
#
# Important:
# - No pixels or patches are deleted.
# - Minimum-size rule is a DISPLAY FLAG ONLY.
# - All calculations use unfiltered raster pixels.
# - Outputs are NOT surveyed/legal wetland boundaries.
# ============================================================


ROOT = Path(__file__).resolve().parents[2]

MAP_DIR = (
    ROOT / "data/processed/part3_future_projection"
)

METRIC_DIR = (
    ROOT / "outputs/metrics/part3_future_projection"
)

REPORT_DIR = (
    ROOT / "outputs/reports/part3_future_projection"
)

RASTER_2025 = (
    MAP_DIR / "LULC_2025_6Class_CommonMask.tif"
)

RASTER_2030 = (
    MAP_DIR / "LULC_2030_Exploratory_CA_Markov.tif"
)

WETLAND_CLASS = 1

# Eight-neighbour connectivity:
# pixels touching along an edge or corner belong
# to the same connected patch.
CONNECTIVITY = 8

# Provisional QGIS display threshold only.
# Five Landsat pixels = 0.45 hectares.
DISPLAY_MIN_PIXELS = 5

EXPECTED_CRS = "EPSG:32644"
EXPECTED_COMMON_PIXELS = 94258

OUTPUTS = {
    "wetland_2025": (
        "Wetland_2025_Estimated_Patches.gpkg"
    ),
    "wetland_2030": (
        "Wetland_2030_Exploratory_Patches.gpkg"
    ),
    "persistence": (
        "Wetland_2025_2030_Persistence.gpkg"
    ),
    "scenario_loss": (
        "Wetland_2025_2030_Scenario_Loss.gpkg"
    ),
    "scenario_gain": (
        "Wetland_2025_2030_Scenario_Gain.gpkg"
    ),
}


def load_rasters():
    rasters = {}
    reference_grid = None

    for year, path in [
        (2025, RASTER_2025),
        (2030, RASTER_2030),
    ]:

        if not path.is_file():
            raise FileNotFoundError(path)

        with rasterio.open(path) as src:

            if (
                str(src.crs) != EXPECTED_CRS
                or src.count != 1
                or src.nodata != 0
            ):
                raise ValueError(
                    f"{year}: unexpected raster metadata."
                )

            grid = (
                src.crs,
                src.transform,
                src.width,
                src.height,
            )

            if reference_grid is None:
                reference_grid = grid
            elif grid != reference_grid:
                raise ValueError(
                    "2025 and 2030 raster grids differ."
                )

            rasters[year] = src.read(1)

    current = rasters[2025]
    future = rasters[2030]

    valid = current != 0

    if not np.array_equal(
        valid,
        future != 0
    ):
        raise ValueError(
            "2025 and 2030 common masks differ."
        )

    if int(valid.sum()) != EXPECTED_COMMON_PIXELS:
        raise ValueError(
            "Unexpected common study-area pixel count."
        )

    for year, array in rasters.items():

        if not np.isin(
            array[valid],
            [1, 2, 3, 4, 5, 6]
        ).all():
            raise ValueError(
                f"{year}: invalid LULC class found."
            )

    crs, transform, width, height = reference_grid

    pixel_area_m2 = abs(
        transform.a * transform.e
        - transform.b * transform.d
    )

    if not np.isclose(
        pixel_area_m2,
        900.0,
        atol=1e-5
    ):
        raise ValueError(
            "Expected a 30 m x 30 m pixel grid."
        )

    return (
        current,
        future,
        valid,
        crs,
        transform,
        pixel_area_m2,
    )


def retain_polygonal_geometry(geometry):
    """
    GDAL eight-connected polygonization can create
    self-touching rings at diagonal connections.

    Preserve all polygonal parts after validity repair.
    """

    if geometry.is_empty:
        raise ValueError(
            "Polygonization produced empty geometry."
        )

    if not geometry.is_valid:
        geometry = make_valid(geometry)

    if geometry.geom_type in (
        "Polygon",
        "MultiPolygon"
    ):
        return geometry

    polygon_parts = []

    def collect_parts(item):

        if item.geom_type == "Polygon":
            polygon_parts.append(item)

        elif item.geom_type == "MultiPolygon":
            polygon_parts.extend(
                list(item.geoms)
            )

        elif item.geom_type == "GeometryCollection":
            for part in item.geoms:
                collect_parts(part)

    collect_parts(geometry)

    if not polygon_parts:
        raise ValueError(
            "No polygonal parts after geometry repair."
        )

    repaired = unary_union(polygon_parts)

    if (
        repaired.is_empty
        or not repaired.is_valid
        or repaired.geom_type not in (
            "Polygon",
            "MultiPolygon"
        )
    ):
        raise ValueError(
            "Could not obtain valid polygonal geometry."
        )

    return repaired


def extract_patches(
    binary_mask,
    layer_name,
    crs,
    transform,
    pixel_area_m2,
):
    """
    Extract every eight-connected raster patch.

    Raster patch labels are polygonized individually,
    and polygon areas must reproduce raster areas.
    """

    binary_mask = np.asarray(
        binary_mask,
        dtype=bool
    )

    structure = np.ones(
        (3, 3),
        dtype=np.uint8
    )

    labels, patch_count = label(
        binary_mask.astype(np.uint8),
        structure=structure
    )

    labels = labels.astype(np.int32)

    sizes = np.bincount(
        labels[binary_mask],
        minlength=patch_count + 1
    )

    seen_labels = set()
    records = []

    for geometry_data, value in shapes(
        labels,
        mask=binary_mask,
        transform=transform,
        connectivity=CONNECTIVITY,
    ):

        patch_number = int(value)

        if patch_number <= 0:
            raise ValueError(
                "Unexpected background polygon."
            )

        if patch_number in seen_labels:
            raise ValueError(
                "A raster patch was polygonized more "
                "than once. Review connectivity handling."
            )

        seen_labels.add(patch_number)

        pixel_count = int(
            sizes[patch_number]
        )

        expected_area_m2 = (
            pixel_count * pixel_area_m2
        )

        geometry = retain_polygonal_geometry(
            shape(geometry_data)
        )

        # Check that vectorization preserved the
        # area represented by the source pixels.
        tolerance = max(
            0.05,
            expected_area_m2 * 1e-9
        )

        if not np.isclose(
            geometry.area,
            expected_area_m2,
            rtol=0,
            atol=tolerance
        ):
            raise ValueError(
                f"{layer_name}: raster/vector area "
                f"mismatch for patch {patch_number}. "
                f"Raster={expected_area_m2:.3f} m2; "
                f"vector={geometry.area:.3f} m2."
            )

        records.append({
            "patch_id": (
                f"{layer_name}_{patch_number:05d}"
            ),
            "patch_no": patch_number,
            "patch_type": layer_name,
            "pixel_count": pixel_count,
            "area_ha": round(
                expected_area_m2 / 10000,
                6
            ),
            "display_ge5": int(
                pixel_count >= DISPLAY_MIN_PIXELS
            ),
            "source_type": "MODEL_SCREENING",
            "geometry": geometry,
        })

    if len(records) != patch_count:
        raise ValueError(
            f"{layer_name}: patch count mismatch. "
            f"Raster={patch_count}, "
            f"vector={len(records)}."
        )

    if len(seen_labels) != patch_count:
        raise ValueError(
            f"{layer_name}: some patches were omitted."
        )

    if not records:
        raise ValueError(
            f"{layer_name}: no polygon patches found."
        )

    gdf = gpd.GeoDataFrame(
        records,
        geometry="geometry",
        crs=crs
    )

    if not gdf.geometry.is_valid.all():
        raise ValueError(
            f"{layer_name}: invalid output geometries."
        )

    if int(gdf["pixel_count"].sum()) != int(
        binary_mask.sum()
    ):
        raise ValueError(
            f"{layer_name}: total pixel accounting failed."
        )

    actual_vector_area = float(
        gdf.geometry.area.sum()
    )

    expected_total_area = (
        int(binary_mask.sum()) * pixel_area_m2
    )

    if not np.isclose(
        actual_vector_area,
        expected_total_area,
        rtol=0,
        atol=0.1
    ):
        raise ValueError(
            f"{layer_name}: total vector area differs "
            "from total raster area."
        )

    return gdf


def main():

    print("=" * 70)
    print("PART 3 - FUTURE WETLAND BOUNDARY PROJECTION")
    print("SCRIPT 09 - WETLAND POLYGON EXTRACTION")
    print("=" * 70)

    (
        current,
        future,
        valid,
        crs,
        transform,
        pixel_area_m2,
    ) = load_rasters()

    current_wetland = (
        (current == WETLAND_CLASS) & valid
    )

    future_wetland = (
        (future == WETLAND_CLASS) & valid
    )

    persistence = (
        current_wetland & future_wetland
    )

    loss = (
        current_wetland & ~future_wetland
    )

    gain = (
        ~current_wetland & future_wetland & valid
    )

    print("\nINPUT CHECKS: PASS")
    print("CRS:", crs)
    print("Common pixels:", int(valid.sum()))
    print("Pixel area:", pixel_area_m2, "m2")

    # --------------------------------------------------------
    # 1. VERIFY RASTER CHANGE ACCOUNTING
    # --------------------------------------------------------

    current_pixels = int(
        current_wetland.sum()
    )

    future_pixels = int(
        future_wetland.sum()
    )

    persistence_pixels = int(
        persistence.sum()
    )

    loss_pixels = int(loss.sum())
    gain_pixels = int(gain.sum())

    if (
        current_pixels
        != persistence_pixels + loss_pixels
    ):
        raise ValueError(
            "2025 wetland loss accounting failed."
        )

    if (
        future_pixels
        != persistence_pixels + gain_pixels
    ):
        raise ValueError(
            "2030 wetland gain accounting failed."
        )

    if (
        future_pixels - current_pixels
        != gain_pixels - loss_pixels
    ):
        raise ValueError(
            "Net wetland change accounting failed."
        )

    print("\nRASTER AREA ACCOUNTING")
    print("-" * 65)

    print(
        "2025 estimated wetland pixels:",
        current_pixels
    )

    print(
        "2030 scenario wetland pixels:",
        future_pixels
    )

    print(
        "Scenario persistent wetland pixels:",
        persistence_pixels
    )

    print(
        "Scenario wetland loss pixels:",
        loss_pixels
    )

    print(
        "Scenario wetland gain pixels:",
        gain_pixels
    )

    print("Raster change accounting: PASS")

    # --------------------------------------------------------
    # 2. EXTRACT ALL PATCHES WITHOUT FILTERING
    # --------------------------------------------------------

    mask_definitions = {
        "wetland_2025": current_wetland,
        "wetland_2030": future_wetland,
        "persistence": persistence,
        "scenario_loss": loss,
        "scenario_gain": gain,
    }

    MAP_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    METRIC_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    summaries = []
    output_files = {}

    print("\nEXTRACTING ALL WETLAND PATCHES")
    print("-" * 65)

    for layer_name, mask in mask_definitions.items():

        print(f"\nProcessing: {layer_name}")

        gdf = extract_patches(
            mask,
            layer_name,
            crs,
            transform,
            pixel_area_m2,
        )

        path = (
            MAP_DIR / OUTPUTS[layer_name]
        )

        # Each GeoPackage is independent, avoiding
        # accidental replacement of other layers.
        if path.exists():
            path.unlink()

        gdf.to_file(
            path,
            layer=layer_name,
            driver="GPKG",
            index=False,
        )

        # Verify the saved GeoPackage is readable
        # and retains all expected features.
        saved = gpd.read_file(
            path,
            layer=layer_name,
        )

        if (
            len(saved) != len(gdf)
            or int(saved["pixel_count"].sum())
            != int(mask.sum())
        ):
            raise ValueError(
                f"{layer_name}: saved GeoPackage "
                "failed the read-back check."
            )

        all_pixels = int(
            gdf["pixel_count"].sum()
        )

        display_gdf = gdf[
            gdf["display_ge5"] == 1
        ]

        small_gdf = gdf[
            gdf["display_ge5"] == 0
        ]

        all_area_ha = (
            all_pixels * pixel_area_m2 / 10000
        )

        small_area_ha = (
            int(small_gdf["pixel_count"].sum())
            * pixel_area_m2 / 10000
        )

        summaries.append({
            "layer": layer_name,
            "patch_count": len(gdf),
            "all_pixels": all_pixels,
            "all_area_ha": round(
                all_area_ha, 4
            ),
            "display_ge5_patch_count":
                len(display_gdf),
            "small_patch_count":
                len(small_gdf),
            "small_patch_area_ha": round(
                small_area_ha, 4
            ),
            "largest_patch_ha": round(
                float(gdf["area_ha"].max()),
                4
            ),
        })

        output_files[layer_name] = str(path)

        print("  Total patches:", len(gdf))
        print("  Total pixels:", all_pixels)
        print(
            "  Total area:",
            f"{all_area_ha:.2f} ha"
        )
        print(
            "  Display-flagged patches (>=5 pixels):",
            len(display_gdf)
        )
        print(
            "  Smaller patches retained:",
            len(small_gdf)
        )
        print("  Raster/vector area check: PASS")
        print("  GeoPackage read-back check: PASS")
        print("  Saved:", path.name)

    # --------------------------------------------------------
    # 3. SAVE IMPORTANT RESEARCH SUMMARIES
    # --------------------------------------------------------

    summary = pd.DataFrame(
        summaries
    )

    summary_path = (
        METRIC_DIR
        / "script09_wetland_patch_summary.csv"
    )

    metadata_path = (
        REPORT_DIR
        / "script09_wetland_polygon_screening.json"
    )

    summary.to_csv(
        summary_path,
        index=False
    )

    report = {
        "component": (
            "Part 3 - Future Wetland Boundary Projection"
        ),
        "stage": (
            "Historical and 2030 scenario polygon extraction"
        ),
        "generated_utc":
            datetime.now(timezone.utc).isoformat(),
        "status": (
            "VECTOR_EXTRACTION_COMPLETE_"
            "SCENARIO_UNVALIDATED"
        ),
        "input_rasters": {
            "2025": str(RASTER_2025),
            "2030": str(RASTER_2030),
        },
        "spatial_reference": str(crs),
        "connectivity": CONNECTIVITY,
        "common_valid_pixels": int(
            valid.sum()
        ),
        "pixel_area_m2": pixel_area_m2,
        "display_min_pixels": DISPLAY_MIN_PIXELS,
        "display_min_area_ha": (
            DISPLAY_MIN_PIXELS
            * pixel_area_m2 / 10000
        ),
        "small_patches_removed": False,
        "analytical_area_filtering_applied": False,
        "scenario_persistence_pixels":
            persistence_pixels,
        "scenario_loss_pixels":
            loss_pixels,
        "scenario_gain_pixels":
            gain_pixels,
        "net_scenario_change_pixels": (
            gain_pixels - loss_pixels
        ),
        "outputs": output_files,
        "validation": {
            "common_raster_mask": True,
            "raster_change_accounting": True,
            "all_connected_patches_preserved": True,
            "raster_to_vector_area_preserved": True,
            "saved_geopackages_readable": True,
        },
        "limitations": [
            "2025 wetland patches are derived from "
            "a model-classified LULC raster.",
            "2030 patches are an exploratory scenario, "
            "not validated future wetland boundaries.",
            "Loss and gain indicate simulated changes, "
            "not confirmed environmental transitions.",
            "The five-pixel display threshold is a "
            "provisional cartographic aid only.",
            "All small patches remain in the analysis.",
            "Eight-neighbour connectivity joins cells "
            "that meet at edges or corners.",
            "Polygon edges follow 30-metre raster cells "
            "and do not represent surveyed boundaries.",
            "No legal conservation status is implied.",
        ],
    }

    metadata_path.write_text(
        json.dumps(
            report,
            indent=2
        ),
        encoding="utf-8"
    )

    print("\n" + "=" * 70)
    print("SCRIPT 09 - FINAL SUMMARY")
    print("=" * 70)

    print(
        summary[
            [
                "layer",
                "patch_count",
                "all_area_ha",
                "small_patch_area_ha",
            ]
        ].to_string(index=False)
    )

    print("\nSaved summary:", summary_path)
    print("Saved metadata:", metadata_path)

    print("\nSCRIPT 09 POLYGON EXTRACTION: COMPLETED")
    print("RASTER-TO-VECTOR AREA CHECKS: PASSED")
    print("SMALL PATCHES: PRESERVED")
    print("GN-WISE SCENARIO REPORTING: NOT YET STARTED")
    print("2030 FORECAST VALIDATION: NOT ESTABLISHED")


if __name__ == "__main__":
    main()
