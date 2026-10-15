
from pathlib import Path
from datetime import datetime, timezone
import json

import geopandas as gpd
import joblib
import numpy as np
import pandas as pd
import rasterio

from rasterio.features import geometry_mask
from shapely.geometry import mapping


# ============================================================
# PART 3 - FUTURE WETLAND BOUNDARY PROJECTION
# SCRIPT 05 - GENERATE HISTORICAL LULC MAPS
#
# Uses the frozen, held-out-evaluated XGBoost classifier.
#
# Outputs:
#   1. Six-class LULC rasters for 2015, 2020, 2025
#   2. Wetland probability rasters for each year
#   3. Class-area summary and processing metadata
#
# Important:
#   - Identical 30 m grid for all years
#   - Identical valid-pixel mask for all years
#   - GN boundary study-area clipping
#   - No model retraining or new model selection
# ============================================================


ROOT = Path(__file__).resolve().parents[2]

LANDSAT_DIR = ROOT / "data/raw/landsat"

BOUNDARY_FILE = (
    ROOT / "data/raw/boundaries/Kaduwela TO Area.kml"
)

GN_LAYER = "GN Boundary Colombo District (#2)"

MODEL_FILE = (
    ROOT
    / "models/part3_future_projection"
    / "lulc_xgb_weighted_final_candidate.joblib"
)

OUTPUT_DIR = (
    ROOT / "data/processed/part3_future_projection"
)

METRIC_DIR = (
    ROOT / "outputs/metrics/part3_future_projection"
)

REPORT_DIR = (
    ROOT / "outputs/reports/part3_future_projection"
)

YEARS = [2015, 2020, 2025]

FEATURES = [
    "blue",
    "green",
    "red",
    "nir",
    "swir1",
    "swir2",
    "ndvi",
    "mndwi",
    "ndbi",
]

CLASS_NAMES = {
    1: "Wetland_Marsh",
    2: "Open_Water",
    3: "Built_Up",
    4: "Paddy_Agriculture",
    5: "Other_Vegetation",
    6: "Bare_Fill_Disturbed",
}

# GIS class 0 represents NoData, not a land-cover class.
NODATA_CLASS = 0

# Floating-point NoData for probability rasters.
NODATA_PROBABILITY = -9999.0

# Pixel-centre rasterization:
# include pixels whose centres fall inside GN polygons.
ALL_TOUCHED = False

# Color scheme for viewing categorical maps in QGIS.
CLASS_COLORS = {
    0: (0, 0, 0, 0),
    1: (48, 125, 93, 255),
    2: (45, 125, 211, 255),
    3: (195, 80, 75, 255),
    4: (225, 190, 80, 255),
    5: (115, 174, 78, 255),
    6: (171, 141, 106, 255),
}


def load_gn_mask(crs, transform, height, width):
    """
    Read the 56-GN study-area boundary and rasterize
    it onto the Landsat grid.

    This does not modify the original KML.
    """

    if not BOUNDARY_FILE.is_file():
        raise FileNotFoundError(BOUNDARY_FILE)

    gn = gpd.read_file(
        BOUNDARY_FILE,
        layer=GN_LAYER,
    )

    if len(gn) != 56:
        raise ValueError(
            f"Expected 56 GN polygons; found {len(gn)}."
        )

    if gn.crs is None:
        raise ValueError(
            "GN boundary has no declared CRS."
        )

    gn = gn.to_crs(crs)

    geometries = []

    for geometry in gn.geometry:

        if geometry is None or geometry.is_empty:
            raise ValueError(
                "GN boundary contains missing geometry."
            )

        if not geometry.is_valid:
            geometry = geometry.buffer(0)

        if (
            geometry.is_empty
            or not geometry.is_valid
            or geometry.geom_type not in [
                "Polygon",
                "MultiPolygon",
            ]
        ):
            raise ValueError(
                "GN boundary contains unusable geometry."
            )

        geometries.append(
            mapping(geometry)
        )

    mask = geometry_mask(
        geometries,
        out_shape=(height, width),
        transform=transform,
        invert=True,
        all_touched=ALL_TOUCHED,
    )

    return mask, len(geometries)


def load_historical_rasters():
    """
    Read the three historical Landsat rasters and
    independently validate their alignment.

    The rasters must have identical:
    - CRS
    - Dimensions
    - Pixel grid / affine transform
    - Nine-band feature ordering
    """

    raster_data = {}
    raster_valid_masks = {}
    raster_details = {}

    reference_grid = None
    reference_profile = None

    print("\nLOADING LANDSAT RASTERS")
    print("-" * 65)

    for year in YEARS:

        path = (
            LANDSAT_DIR
            / f"Landsat_{year}_KMC_9Band.tif"
        )

        if not path.is_file():
            raise FileNotFoundError(path)

        with rasterio.open(path) as src:

            if str(src.crs) != "EPSG:32644":
                raise ValueError(
                    f"{year}: unexpected CRS {src.crs}"
                )

            if src.count != len(FEATURES):
                raise ValueError(
                    f"{year}: expected nine spectral bands."
                )

            if src.res != (30.0, 30.0):
                raise ValueError(
                    f"{year}: expected 30 m resolution."
                )

            actual_descriptions = [
                (description or "").strip().lower()
                for description in src.descriptions
            ]

            if actual_descriptions != FEATURES:
                raise ValueError(
                    f"{year}: spectral band order differs "
                    f"from the model feature order."
                )

            current_grid = (
                src.crs,
                src.transform,
                src.width,
                src.height,
            )

            if reference_grid is None:

                reference_grid = current_grid
                reference_profile = src.profile.copy()

            elif current_grid != reference_grid:

                raise ValueError(
                    f"{year}: raster grid is not aligned."
                )

            values = src.read(
                out_dtype="float32"
            )

            # Valid only when all nine features are finite
            # and no band contains its NoData value.
            valid = np.all(
                np.isfinite(values),
                axis=0
            )

            if src.nodata is not None:

                valid &= np.all(
                    values != src.nodata,
                    axis=0
                )

            # Honour any GDAL/raster band validity masks.
            band_masks = src.read_masks()

            valid &= np.all(
                band_masks > 0,
                axis=0
            )

            raster_data[year] = values
            raster_valid_masks[year] = valid

            raster_details[str(year)] = {
                "source": str(path),
                "valid_pixels_full_raster":
                    int(valid.sum()),
                "nodata_value": (
                    float(src.nodata)
                    if src.nodata is not None
                    else None
                ),
            }

            print(
                f"{year}: "
                f"{int(valid.sum()):,} valid raster pixels"
            )

    return (
        raster_data,
        raster_valid_masks,
        raster_details,
        reference_grid,
        reference_profile,
    )


def prepare_common_mask(
    raster_valid_masks,
    reference_grid,
):
    """
    Construct one study-area mask shared by all years.

    A pixel is included only if:
    - its centre is inside a Kaduwela GN polygon, AND
    - all nine Landsat features are valid in 2015, AND
    - all nine Landsat features are valid in 2020, AND
    - all nine Landsat features are valid in 2025.
    """

    crs, transform, width, height = reference_grid

    gn_mask, gn_count = load_gn_mask(
        crs,
        transform,
        height,
        width,
    )

    common_valid_mask = gn_mask.copy()

    for year in YEARS:
        common_valid_mask &= raster_valid_masks[year]

    gn_pixels = int(gn_mask.sum())
    common_pixels = int(common_valid_mask.sum())

    if common_pixels == 0:
        raise ValueError(
            "No valid common study-area pixels."
        )

    if common_pixels > gn_pixels:
        raise ValueError(
            "Common mask unexpectedly exceeds GN boundary."
        )

    print("\nCOMMON STUDY-AREA MASK")
    print("-" * 65)
    print("GN polygons:", gn_count)
    print("Pixels inside GN study area:", gn_pixels)
    print("Common valid pixels:", common_pixels)
    print(
        "Excluded GN pixels due to missing year/band:",
        gn_pixels - common_pixels
    )

    pixel_area_m2 = abs(
        transform.a * transform.e
        - transform.b * transform.d
    )

    pixel_area_ha = pixel_area_m2 / 10000.0

    print(
        "Common valid area:",
        f"{common_pixels * pixel_area_ha:.2f} ha"
    )

    return (
        common_valid_mask,
        gn_mask,
        pixel_area_ha,
    )


def predict_year(
    model,
    values,
    common_mask,
):
    """
    Predict six LULC classes for pixels in the
    common mask, using the frozen XGBoost model.

    XGBoost model classes: 0,1,2,3,4,5
    GIS research classes: 1,2,3,4,5,6
    """

    height, width = common_mask.shape

    pixels = (
        values[:, common_mask]
        .T.astype("float32")
    )

    if pixels.shape[1] != 9:
        raise ValueError(
            "Incorrect number of features."
        )

    if not np.isfinite(pixels).all():
        raise ValueError(
            "Invalid spectral features in common mask."
        )

    model_classes = np.asarray(
        model.classes_,
        dtype=int,
    )

    if not np.array_equal(
        model_classes,
        np.arange(6),
    ):
        raise ValueError(
            "Model class encoding differs from 0-5."
        )

    raw_predictions = model.predict(
        pixels
    ).astype(int)

    probabilities = model.predict_proba(
        pixels
    )

    if probabilities.shape != (len(pixels), 6):
        raise ValueError(
            "Unexpected classification probability shape."
        )

    if (
        not np.isfinite(probabilities).all()
        or np.any(probabilities < -1e-7)
        or np.any(probabilities > 1 + 1e-7)
    ):
        raise ValueError(
            "Invalid classification probabilities."
        )

    if not np.allclose(
        probabilities.sum(axis=1),
        1.0,
        atol=1e-5,
    ):
        raise ValueError(
            "Class probabilities do not sum to one."
        )

    if not np.array_equal(
        raw_predictions,
        np.argmax(probabilities, axis=1),
    ):
        raise ValueError(
            "Predicted labels disagree with probabilities."
        )

    # Convert zero-based model classes to GIS class IDs.
    predicted_classes = (
        raw_predictions + 1
    ).astype("uint8")

    lulc = np.full(
        (height, width),
        NODATA_CLASS,
        dtype="uint8",
    )

    wetland_probability = np.full(
        (height, width),
        NODATA_PROBABILITY,
        dtype="float32",
    )

    maximum_probability = np.full(
        (height, width),
        NODATA_PROBABILITY,
        dtype="float32",
    )

    lulc[common_mask] = predicted_classes

    # Class 0 in the model represents Wetland/Marsh.
    wetland_probability[common_mask] = (
        probabilities[:, 0]
    )

    maximum_probability[common_mask] = (
        probabilities.max(axis=1)
    )

    return (
        lulc,
        wetland_probability,
        maximum_probability,
    )


def write_lulc(
    path,
    array,
    reference_profile,
    year,
):
    profile = reference_profile.copy()

    profile.update(
        driver="GTiff",
        count=1,
        dtype="uint8",
        nodata=NODATA_CLASS,
        compress="deflate",
    )

    with rasterio.open(
        path,
        "w",
        **profile,
    ) as dst:

        dst.write(array, 1)

        dst.set_band_description(
            1,
            f"Historical LULC {year}",
        )

        dst.write_colormap(
            1,
            CLASS_COLORS,
        )

        dst.update_tags(
            year=str(year),
            classification_method=(
                "Frozen XGBoost multiclass classifier"
            ),
            mapping=(
                "1=Wetland_Marsh;"
                "2=Open_Water;"
                "3=Built_Up;"
                "4=Paddy_Agriculture;"
                "5=Other_Vegetation;"
                "6=Bare_Fill_Disturbed"
            ),
            validity_mask=(
                "Kaduwela 56 GN polygons and "
                "shared valid 2015/2020/2025 pixels"
            ),
        )


def write_probability(
    path,
    array,
    reference_profile,
    year,
):
    profile = reference_profile.copy()

    profile.update(
        driver="GTiff",
        count=1,
        dtype="float32",
        nodata=NODATA_PROBABILITY,
        compress="deflate",
        predictor=3,
    )

    with rasterio.open(
        path,
        "w",
        **profile,
    ) as dst:

        dst.write(array, 1)

        dst.set_band_description(
            1,
            f"Wetland model probability {year}",
        )

        dst.update_tags(
            year=str(year),
            probability_class="1_Wetland_Marsh",
            probability_range="0_to_1",
            interpretation=(
                "Model-estimated probability, "
                "not calibrated confidence or surveyed truth"
            ),
        )


def main():

    print("=" * 70)
    print("PART 3 - FUTURE WETLAND BOUNDARY PROJECTION")
    print("SCRIPT 05 - HISTORICAL LULC MAP GENERATION")
    print("=" * 70)

    if not MODEL_FILE.is_file():
        raise FileNotFoundError(
            f"Frozen model not found: {MODEL_FILE}"
        )

    model = joblib.load(MODEL_FILE)

    print("\nFROZEN CLASSIFIER")
    print("-" * 65)
    print("Model:", type(model).__name__)
    print("Model file:", MODEL_FILE)
    print("Model classes:", model.classes_)

    (
        raster_data,
        valid_masks,
        raster_details,
        reference_grid,
        reference_profile,
    ) = load_historical_rasters()

    (
        common_mask,
        gn_mask,
        pixel_area_ha,
    ) = prepare_common_mask(
        valid_masks,
        reference_grid,
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    METRIC_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    area_rows = []
    year_summaries = {}

    print("\nGENERATING HISTORICAL CLASSIFICATIONS")
    print("-" * 65)

    for year in YEARS:

        print(f"\nYEAR {year}")

        (
            lulc,
            wetland_probability,
            maximum_probability,
        ) = predict_year(
            model,
            raster_data[year],
            common_mask,
        )

        # Fundamental output checks.
        if not np.array_equal(
            lulc != NODATA_CLASS,
            common_mask,
        ):
            raise ValueError(
                f"{year}: output does not match common mask."
            )

        predicted_values = lulc[common_mask]

        if not np.isin(
            predicted_values,
            list(CLASS_NAMES),
        ).all():
            raise ValueError(
                f"{year}: invalid LULC classes."
            )

        lulc_path = (
            OUTPUT_DIR
            / f"LULC_{year}_6Class_CommonMask.tif"
        )

        wetland_path = (
            OUTPUT_DIR
            / f"Wetland_Probability_{year}_CommonMask.tif"
        )

        write_lulc(
            lulc_path,
            lulc,
            reference_profile,
            year,
        )

        write_probability(
            wetland_path,
            wetland_probability,
            reference_profile,
            year,
        )

        for class_id, class_name in CLASS_NAMES.items():

            count = int(
                (predicted_values == class_id).sum()
            )

            area_ha = count * pixel_area_ha

            area_rows.append({
                "year": year,
                "class_id": class_id,
                "class_name": class_name,
                "pixel_count": count,
                "area_ha": round(area_ha, 4),
            })

            print(
                f"  {class_id}. "
                f"{class_name:22s} "
                f"{count:7,d} pixels | "
                f"{area_ha:9.2f} ha"
            )

        valid_max_probability = (
            maximum_probability[common_mask]
        )

        wetland_pixels = int(
            (predicted_values == 1).sum()
        )

        year_summaries[str(year)] = {
            "lulc_raster": str(lulc_path),
            "wetland_probability_raster": str(
                wetland_path
            ),
            "classified_pixels": int(
                common_mask.sum()
            ),
            "predicted_wetland_pixels":
                wetland_pixels,
            "predicted_wetland_area_ha": round(
                wetland_pixels * pixel_area_ha,
                4,
            ),
            "mean_max_class_probability": round(
                float(valid_max_probability.mean()),
                6,
            ),
            "fraction_max_probability_below_0_5": round(
                float(
                    (valid_max_probability < 0.5).mean()
                ),
                6,
            ),
        }

        print(
            "  Mean highest class probability:",
            f"{valid_max_probability.mean():.4f}"
        )

        print(
            "  Fraction with highest probability < 0.5:",
            f"{(valid_max_probability < 0.5).mean():.4f}"
        )

        print("  Saved:", lulc_path.name)
        print("  Saved:", wetland_path.name)

    # --------------------------------------------------------
    # CROSS-YEAR CONSISTENCY CHECK
    # --------------------------------------------------------

    summary = pd.DataFrame(area_rows)

    common_pixels = int(common_mask.sum())

    for year in YEARS:

        total = int(
            summary.loc[
                summary["year"] == year,
                "pixel_count",
            ].sum()
        )

        if total != common_pixels:
            raise ValueError(
                f"{year}: classified pixel accounting failed."
            )

    area_path = (
        METRIC_DIR
        / "script05_historical_lulc_area_summary.csv"
    )

    summary_path = (
        REPORT_DIR
        / "script05_historical_map_generation_summary.json"
    )

    summary.to_csv(
        area_path,
        index=False,
    )

    report = {
        "component": (
            "Part 3 - Future Wetland Boundary Projection"
        ),
        "stage": "Historical six-class LULC maps",
        "generated_utc":
            datetime.now(timezone.utc).isoformat(),
        "model_source": str(MODEL_FILE),
        "model_status": (
            "Frozen evaluated classifier; "
            "historical maps remain estimates"
        ),
        "years": YEARS,
        "features_in_order": FEATURES,
        "class_ids": {
            str(k): v
            for k, v in CLASS_NAMES.items()
        },
        "land_cover_nodata_class": NODATA_CLASS,
        "gn_boundary_source": str(BOUNDARY_FILE),
        "gn_layer": GN_LAYER,
        "gn_polygon_count": 56,
        "spatial_reference": str(reference_grid[0]),
        "resolution_m": 30,
        "pixel_area_ha": pixel_area_ha,
        "pixels_inside_gn_boundary": int(
            gn_mask.sum()
        ),
        "common_valid_pixels": common_pixels,
        "common_valid_area_ha": round(
            common_pixels * pixel_area_ha,
            4,
        ),
        "common_mask_definition": (
            "Pixel centre inside one of 56 GN polygons "
            "and all nine spectral features valid "
            "in every historical year"
        ),
        "input_rasters": raster_details,
        "historical_outputs": year_summaries,
        "validation": {
            "aligned_input_grids": True,
            "same_common_mask_all_years": True,
            "six_class_encoding_verified": True,
            "classified_pixel_accounting": True,
            "model_not_retrained": True,
        },
        "limitations": [
            "Maps are estimated land cover, not field-surveyed truth.",
            "Held-out wetland recall was limited, particularly in 2015.",
            "Wetland and paddy classification errors can affect "
            "subsequent estimated transition rates.",
            "Model probabilities are not independently calibrated.",
            "The 2025 map is an estimated comparison map "
            "for hindcast analysis, not error-free ground truth.",
            "NoData differences are controlled using a fixed "
            "three-year common spatial mask.",
        ],
    }

    summary_path.write_text(
        json.dumps(
            report,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\n" + "=" * 70)
    print("SCRIPT 05 - FINAL SUMMARY")
    print("=" * 70)

    print("Historical LULC maps generated:", len(YEARS))
    print("Wetland probability maps generated:", len(YEARS))
    print("Common classified pixels per year:", common_pixels)

    print("\nPREDICTED WETLAND AREA BY YEAR")

    for year in YEARS:
        print(
            f"{year}: "
            f"{year_summaries[str(year)]['predicted_wetland_area_ha']:.2f} ha"
        )

    print("\nSaved class-area summary:", area_path)
    print("Saved generation metadata:", summary_path)

    print("\nSCRIPT 05 MAP GENERATION: COMPLETED")
    print("HISTORICAL CHANGE INTERPRETATION: NOT YET VALIDATED")
    print("CA-MARKOV TRANSITION ANALYSIS: NOT YET STARTED")


if __name__ == "__main__":
    main()
