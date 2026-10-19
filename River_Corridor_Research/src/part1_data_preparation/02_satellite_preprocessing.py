
"""
J26-DS-338 - River Corridor Research

STEP 10: LANDSAT PREPROCESSING AND SPECTRAL INDICES

Inputs:
    18 Landsat GeoTIFF images (2010-2025)
    Candidate river corridor GeoJSON

Outputs:
    3-band index GeoTIFF per observation:
        Band 1: MNDWI
        Band 2: NDVI
        Band 3: NDBI

    CSV containing image-level quality measurements.

Important:
    Landsat reflectance bands were already scaled
    during Google Earth Engine export.

    Landsat 7 primary and supplementary observations
    remain separate. No dates are merged.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio

from rasterio.features import geometry_mask


# ==================================================
# 1. PROJECT PATHS
# ==================================================

ROOT = Path(__file__).resolve().parents[2]

INPUT_DIR = ROOT / "data" / "raw" / "landsat"

CORRIDOR_PATH = (
    ROOT / "data" / "raw" / "boundaries"
    / "kelani_river_corridor.geojson"
)

OUTPUT_DIR = (
    ROOT / "data" / "processed"
    / "landsat_indices"
)

REPORT_DIR = ROOT / "outputs" / "reports"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

NODATA = -9999.0

EXPECTED_BANDS = (
    "blue", "green", "red", "nir",
    "swir1", "swir2",
    "qa_pixel", "qa_radsat", "valid_pixel"
)


# ==================================================
# 2. SPECTRAL INDEX CALCULATION
# ==================================================

def calculate_index(band_a, band_b, valid_mask):
    """
    Calculate (A - B) / (A + B).

    Prevent division by zero and exclude
    invalid or implausible index values.

    Invalid pixels receive -9999.
    """

    numerator = band_a - band_b
    denominator = band_a + band_b

    usable = (
        valid_mask
        & np.isfinite(numerator)
        & np.isfinite(denominator)
        & (np.abs(denominator) > 1e-6)
    )

    values = np.zeros_like(
        numerator, dtype=np.float32
    )

    np.divide(
        numerator,
        denominator,
        out=values,
        where=usable
    )

    usable = (
        usable
        & np.isfinite(values)
        & (values >= -1.0)
        & (values <= 1.0)
    )

    result = np.full(
        numerator.shape,
        NODATA,
        dtype=np.float32
    )

    result[usable] = values[usable]

    return result


# ==================================================
# 3. LOAD STUDY CORRIDOR
# ==================================================

print("=" * 75)
print("STEP 10 - LANDSAT PREPROCESSING")
print("=" * 75)

if not CORRIDOR_PATH.exists():
    raise FileNotFoundError(
        f"Missing river corridor: {CORRIDOR_PATH}"
    )

corridor = gpd.read_file(CORRIDOR_PATH)

if corridor.empty or corridor.crs is None:
    raise ValueError(
        "River corridor geometry or CRS is invalid."
    )

landsat_files = sorted(INPUT_DIR.glob("*.tif"))

print(f"\nLandsat files found: {len(landsat_files)}")

if len(landsat_files) != 18:
    raise ValueError(
        "Expected 18 Landsat observations. "
        "Please verify the raw files."
    )

print("Study corridor loaded successfully.")

records = []
reference_grid = None


# ==================================================
# 4. PROCESS LANDSAT OBSERVATIONS
# ==================================================

for number, file in enumerate(landsat_files, 1):

    print("\n" + "-" * 75)
    print(f"IMAGE {number}/{len(landsat_files)}")
    print(f"Filename: {file.name}")

    with rasterio.open(file) as src:

        # ------------------------------------------
        # A. VALIDATE IMAGE STRUCTURE
        # ------------------------------------------

        if src.count != 9:
            raise ValueError(
                f"{file.name}: Expected 9 bands, "
                f"found {src.count}"
            )

        if src.crs is None or src.crs.to_epsg() != 32644:
            raise ValueError(
                f"{file.name}: Unexpected CRS {src.crs}"
            )

        if not np.allclose(
            src.res, (30.0, 30.0)
        ):
            raise ValueError(
                f"{file.name}: Unexpected resolution"
            )

        labels = tuple(
            (name or "").lower()
            for name in src.descriptions
        )

        if any(labels) and labels != EXPECTED_BANDS:
            raise ValueError(
                f"{file.name}: Band names/order "
                f"do not match expected export."
            )

        current_grid = (
            src.width,
            src.height,
            src.transform,
            src.crs
        )

        if reference_grid is None:
            reference_grid = current_grid

        elif current_grid != reference_grid:
            raise ValueError(
                f"{file.name}: Landsat grid mismatch"
            )

        # ------------------------------------------
        # B. PREPARE CORRIDOR MASK
        # ------------------------------------------

        projected = corridor.to_crs(src.crs)

        geometries = [
            geom.__geo_interface__
            for geom in projected.geometry
            if geom is not None and not geom.is_empty
        ]

        if not geometries:
            raise ValueError(
                "No valid corridor geometries."
            )

        corridor_mask = geometry_mask(
            geometries,
            out_shape=(src.height, src.width),
            transform=src.transform,
            invert=True
        )

        corridor_pixels = int(
            corridor_mask.sum()
        )

        if corridor_pixels == 0:
            raise ValueError(
                f"{file.name}: Corridor outside raster"
            )

        # ------------------------------------------
        # C. READ SPECTRAL BANDS
        # ------------------------------------------

        bands = src.read(
            indexes=[1, 2, 3, 4, 5, 6]
        ).astype(np.float32)

        blue, green, red, nir, swir1, swir2 = bands

        qa_pixel = src.read(7)
        qa_radsat = src.read(8)
        valid_pixel = src.read(9)

        # ------------------------------------------
        # D. VALIDITY AND QUALITY MASKS
        # ------------------------------------------

        spectral_valid = np.all(
            np.isfinite(bands)
            & (bands != NODATA),
            axis=0
        )

        source_valid = (
            np.isfinite(valid_pixel)
            & (valid_pixel > 0.5)
        )

        qa_numeric_valid = (
            np.isfinite(qa_pixel)
            & (qa_pixel >= 0)
        )

        qa_integers = np.where(
            qa_numeric_valid,
            qa_pixel,
            0
        ).astype(np.uint32)

        # QA_PIXEL bits 0-5:
        # Fill, dilated cloud, cirrus,
        # cloud, cloud shadow, snow.
        qa_clear = (
            qa_numeric_valid
            & ((qa_integers & 0b111111) == 0)
        )

        saturation_clear = (
            np.isfinite(qa_radsat)
            & (qa_radsat == 0)
        )

        final_valid = (
            corridor_mask
            & spectral_valid
            & source_valid
            & qa_clear
            & saturation_clear
        )

        valid_count = int(final_valid.sum())

        valid_percentage = (
            100.0 * valid_count / corridor_pixels
        )

        print(f"Corridor pixels: {corridor_pixels}")
        print(f"Valid pixels   : {valid_count}")
        print(
            f"Valid coverage : "
            f"{valid_percentage:.2f}%"
        )

        if valid_count == 0:
            raise ValueError(
                f"{file.name}: No usable pixels."
            )

        # ------------------------------------------
        # E. CALCULATE SPECTRAL INDICES
        # ------------------------------------------

        mndwi = calculate_index(
            green, swir1, final_valid
        )

        ndvi = calculate_index(
            nir, red, final_valid
        )

        ndbi = calculate_index(
            swir1, nir, final_valid
        )

        print("MNDWI calculated.")
        print("NDVI calculated.")
        print("NDBI calculated.")

        # ------------------------------------------
        # F. EXPORT INDEX GEOTIFF
        # ------------------------------------------

        output_path = (
            OUTPUT_DIR
            / f"{file.stem}_indices.tif"
        )

        profile = src.profile.copy()

        profile.update(
            driver="GTiff",
            count=3,
            dtype="float32",
            nodata=NODATA,
            compress="deflate",
            predictor=3
        )

        with rasterio.open(
            output_path, "w", **profile
        ) as dst:

            dst.write(mndwi, 1)
            dst.write(ndvi, 2)
            dst.write(ndbi, 3)

            dst.set_band_description(1, "MNDWI")
            dst.set_band_description(2, "NDVI")
            dst.set_band_description(3, "NDBI")

        # ------------------------------------------
        # G. RECORD QA STATISTICS
        # ------------------------------------------

        mndwi_valid = mndwi != NODATA
        ndvi_valid = ndvi != NODATA
        ndbi_valid = ndbi != NODATA

        if mndwi_valid.any():
            mndwi_median = float(
                np.median(mndwi[mndwi_valid])
            )
        else:
            mndwi_median = None

        record = {
            "source_file": file.name,
            "output_file": output_path.name,
            "corridor_pixels": corridor_pixels,
            "valid_pixels": valid_count,
            "valid_percentage": round(
                valid_percentage, 2
            ),
            "mndwi_valid_pixels": int(
                mndwi_valid.sum()
            ),
            "ndvi_valid_pixels": int(
                ndvi_valid.sum()
            ),
            "ndbi_valid_pixels": int(
                ndbi_valid.sum()
            ),
            "median_mndwi": mndwi_median,
            "crs": str(src.crs),
            "resolution_m": 30
        }

        records.append(record)

        print(f"Saved: {output_path.name}")


# ==================================================
# 5. EXPORT PROCESSING QA REPORT
# ==================================================

qa_table = pd.DataFrame(records)

report_path = (
    REPORT_DIR
    / "02_landsat_preprocessing_qa.csv"
)

qa_table.to_csv(
    report_path,
    index=False
)


# ==================================================
# 6. FINAL SUMMARY
# ==================================================

print("\n" + "=" * 75)
print("STEP 10 - PREPROCESSING SUMMARY")
print("=" * 75)

print(f"Images processed: {len(records)}")

print(
    f"Minimum valid coverage: "
    f"{qa_table['valid_percentage'].min():.2f}%"
)

print(
    f"Maximum valid coverage: "
    f"{qa_table['valid_percentage'].max():.2f}%"
)

print("\nOutputs:")
print(OUTPUT_DIR)
print(report_path)

print("\nSTEP 10 FINISHED")
print("=" * 75)
