from pathlib import Path

import pandas as pd
import ee


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

INPUT_CSV = (
    PROCESSED_DIR
    / "historical_wetland_clean_metrics.csv"
)


# ============================================================
# EARTH ENGINE SETTINGS
# ============================================================

EE_PROJECT = "kaduwela-wetland-research"

KMC_ASSET = (
    "projects/kaduwela-wetland-research/"
    "assets/KMC_Boundary"
)

CHIRPS_COLLECTION = (
    "UCSB-CHG/CHIRPS/DAILY"
)


# ============================================================
# INITIALIZE EARTH ENGINE
# ============================================================

ee.Initialize(
    project=EE_PROJECT
)


# ============================================================
# LOAD UNIQUE GOOGLE EARTH OBSERVATION DATES
# ============================================================

metrics = pd.read_csv(
    INPUT_CSV
)

unique_dates = (
    metrics["observation_date"]
    .dropna()
    .drop_duplicates()
    .sort_values()
    .tolist()
)

print(
    f"Unique observation dates: "
    f"{len(unique_dates)}"
)


# ============================================================
# STUDY AREA
# ============================================================

kmc = ee.FeatureCollection(
    KMC_ASSET
)

region = kmc.geometry()


# ============================================================
# CHIRPS DAILY RAINFALL
# ============================================================

chirps = (
    ee.ImageCollection(
        CHIRPS_COLLECTION
    )
    .select("precipitation")
)


# ============================================================
# CREATE ONE FEATURE PER OBSERVATION DATE
# ============================================================

features = []

for date_string in unique_dates:

    observation_date = ee.Date(
        date_string
    )

    # CHIRPS dates are daily.
    # Use the 30 days before and including
    # the observation date.

    end_date = observation_date.advance(
        1,
        "day",
    )

    start_1d = observation_date

    start_7d = observation_date.advance(
        -6,
        "day",
    )

    start_30d = observation_date.advance(
        -29,
        "day",
    )

    # --------------------------------------------------------
    # 1-DAY RAINFALL
    # --------------------------------------------------------

    rainfall_1d = (
        chirps
        .filterDate(
            start_1d,
            end_date,
        )
        .sum()
    )

    # --------------------------------------------------------
    # 7-DAY ANTECEDENT RAINFALL
    # --------------------------------------------------------

    rainfall_7d = (
        chirps
        .filterDate(
            start_7d,
            end_date,
        )
        .sum()
    )

    # --------------------------------------------------------
    # 30-DAY ANTECEDENT RAINFALL
    # --------------------------------------------------------

    rainfall_30d = (
        chirps
        .filterDate(
            start_30d,
            end_date,
        )
        .sum()
    )

    # --------------------------------------------------------
    # KMC AREA-MEAN VALUES
    # --------------------------------------------------------

    rain_1d_mm = (
        rainfall_1d
        .reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=region,
            scale=5566,
            bestEffort=True,
            maxPixels=1e9,
        )
        .get("precipitation")
    )

    rain_7d_mm = (
        rainfall_7d
        .reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=region,
            scale=5566,
            bestEffort=True,
            maxPixels=1e9,
        )
        .get("precipitation")
    )

    rain_30d_mm = (
        rainfall_30d
        .reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=region,
            scale=5566,
            bestEffort=True,
            maxPixels=1e9,
        )
        .get("precipitation")
    )

    feature = ee.Feature(
        None,
        {
            "observation_date":
                date_string,

            "rain_1d_mm":
                rain_1d_mm,

            "rain_7d_mm":
                rain_7d_mm,

            "rain_30d_mm":
                rain_30d_mm,

            "rainfall_source":
                "CHIRPS Daily v2",

            "rainfall_spatial_context":
                "KMC area mean",
        },
    )

    features.append(
        feature
    )


# ============================================================
# EXPORT TO GOOGLE DRIVE
# ============================================================

feature_collection = (
    ee.FeatureCollection(
        features
    )
)

export_task = (
    ee.batch.Export.table.toDrive(
        collection=feature_collection,
        description=(
            "Kaduwela_Historical_"
            "Wetland_Rainfall_Context"
        ),
        folder=(
            "Kaduwela_Wetland_Research"
        ),
        fileNamePrefix=(
            "historical_wetland_"
            "rainfall_context"
        ),
        fileFormat="CSV",
    )
)

export_task.start()

print(
    "\nRainfall export task started."
)

print(
    "Google Drive folder: "
    "Kaduwela_Wetland_Research"
)

print(
    "Expected file:"
)

print(
    "historical_wetland_"
    "rainfall_context.csv"
)

print(
    "\nCheck Earth Engine Tasks / "
    "Google Drive for completion."
)