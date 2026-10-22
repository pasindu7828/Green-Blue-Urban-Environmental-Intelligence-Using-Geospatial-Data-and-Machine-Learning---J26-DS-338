from pathlib import Path

import pandas as pd
import geopandas as gpd


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_RAINFALL_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "rainfall"
)

PROCESSED_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "part1_historical_change"
)

METRICS_INPUT = (
    PROCESSED_DIR
    / "historical_wetland_clean_metrics.csv"
)

SPATIAL_INPUT = (
    PROCESSED_DIR
    / "historical_wetland_clean_observations_utm44n.gpkg"
)

RAINFALL_INPUT = (
    RAW_RAINFALL_DIR
    / "historical_wetland_rainfall_context.csv"
)

OUTPUT_CSV = (
    PROCESSED_DIR
    / "wetland_obs_rainfall.csv"
)

OUTPUT_GPKG = (
    PROCESSED_DIR
    / "wetland_obs_rainfall_utm44n.gpkg"
)

OUTPUT_GEOJSON = (
    PROCESSED_DIR
    / "wetland_obs_rainfall_wgs84.geojson"
)


# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # LOAD DATA
    # --------------------------------------------------------

    metrics = pd.read_csv(
        METRICS_INPUT
    )

    rainfall = pd.read_csv(
        RAINFALL_INPUT
    )

    print("\n========================================")
    print("RAINFALL CONTEXT JOIN")
    print("========================================")

    print(
        f"Wetland observations loaded: "
        f"{len(metrics)}"
    )

    print(
        f"Rainfall-date records loaded: "
        f"{len(rainfall)}"
    )

    # --------------------------------------------------------
    # CLEAN EARTH ENGINE EXPORT COLUMNS
    # --------------------------------------------------------

    rainfall = rainfall.drop(
        columns=[
            "system:index",
            ".geo",
        ],
        errors="ignore",
    )

    # --------------------------------------------------------
    # STANDARDIZE DATE FIELDS
    # --------------------------------------------------------

    metrics["observation_date"] = (
        pd.to_datetime(
            metrics["observation_date"]
        )
        .dt.strftime("%Y-%m-%d")
    )

    rainfall["observation_date"] = (
        pd.to_datetime(
            rainfall["observation_date"]
        )
        .dt.strftime("%Y-%m-%d")
    )

    # --------------------------------------------------------
    # RAINFALL QA
    # --------------------------------------------------------

    duplicate_rainfall_dates = (
        rainfall["observation_date"]
        .duplicated()
        .sum()
    )

    if duplicate_rainfall_dates > 0:
        raise ValueError(
            "Duplicate observation dates found "
            "in rainfall dataset."
        )

    rainfall_columns = [
        "rain_1d_mm",
        "rain_7d_mm",
        "rain_30d_mm",
    ]

    missing_rainfall_values = (
        rainfall[rainfall_columns]
        .isna()
        .sum()
        .sum()
    )

    print(
        f"Duplicate rainfall dates: "
        f"{duplicate_rainfall_dates}"
    )

    print(
        f"Missing rainfall values: "
        f"{missing_rainfall_values}"
    )

    # --------------------------------------------------------
    # JOIN RAINFALL TO OBSERVATION METRICS
    # --------------------------------------------------------

    enriched = metrics.merge(
        rainfall,
        on="observation_date",
        how="left",
        validate="many_to_one",
    )

    unmatched = enriched[
        enriched["rain_30d_mm"].isna()
    ].copy()

    print(
        f"Observations after join: "
        f"{len(enriched)}"
    )

    print(
        f"Observations without rainfall: "
        f"{len(unmatched)}"
    )

    if not unmatched.empty:

        print(
            "\nUnmatched dates:"
        )

        print(
            unmatched[
                [
                    "wetland_id",
                    "observation_date",
                ]
            ]
            .drop_duplicates()
            .to_string(index=False)
        )

        raise RuntimeError(
            "Some wetland observations do not "
            "have rainfall context."
        )

    # --------------------------------------------------------
    # SAVE TABULAR OUTPUT
    # --------------------------------------------------------

    enriched.to_csv(
        OUTPUT_CSV,
        index=False,
    )

    # --------------------------------------------------------
    # JOIN SAME RAINFALL FIELDS TO SPATIAL DATA
    # --------------------------------------------------------

    spatial = gpd.read_file(
        SPATIAL_INPUT,
        layer=(
            "historical_wetland_clean_observations"
        ),
    )

    spatial["observation_date"] = (
        pd.to_datetime(
            spatial["observation_date"]
        )
        .dt.strftime("%Y-%m-%d")
    )

    rainfall_for_spatial = rainfall[
        [
            "observation_date",
            "rain_1d_mm",
            "rain_7d_mm",
            "rain_30d_mm",
            "rainfall_source",
            "rainfall_spatial_context",
        ]
    ].copy()

    spatial_enriched = spatial.merge(
        rainfall_for_spatial,
        on="observation_date",
        how="left",
        validate="many_to_one",
    )

    # --------------------------------------------------------
    # SAVE METRIC CRS SPATIAL OUTPUT
    # --------------------------------------------------------

    spatial_enriched.to_file(
        OUTPUT_GPKG,
        layer=(
            "historical_wetland_observations_rainfall"
        ),
        driver="GPKG",
    )

    # --------------------------------------------------------
    # SAVE WEB / FRONTEND VERSION
    # --------------------------------------------------------

    spatial_web = spatial_enriched.to_crs(
        "EPSG:4326"
    )

    spatial_web.to_file(
        OUTPUT_GEOJSON,
        driver="GeoJSON",
    )

    # --------------------------------------------------------
    # TERMINAL SUMMARY
    # --------------------------------------------------------

    print("\nRainfall ranges:")

    print(
        f"1-day: "
        f"{enriched['rain_1d_mm'].min():.2f} - "
        f"{enriched['rain_1d_mm'].max():.2f} mm"
    )

    print(
        f"7-day: "
        f"{enriched['rain_7d_mm'].min():.2f} - "
        f"{enriched['rain_7d_mm'].max():.2f} mm"
    )

    print(
        f"30-day: "
        f"{enriched['rain_30d_mm'].min():.2f} - "
        f"{enriched['rain_30d_mm'].max():.2f} mm"
    )

    print("\nSeason distribution:")

    print(
        enriched["season"]
        .value_counts()
        .to_string()
    )

    print("\nOutputs:")

    print(OUTPUT_CSV)
    print(OUTPUT_GPKG)
    print(OUTPUT_GEOJSON)

    print(
        "\nNOTE:"
        "\nRainfall values represent CHIRPS "
        "KMC-area mean rainfall context."
        "\nThey should not be interpreted as "
        "point rainfall measured at each wetland."
    )


if __name__ == "__main__":
    main()