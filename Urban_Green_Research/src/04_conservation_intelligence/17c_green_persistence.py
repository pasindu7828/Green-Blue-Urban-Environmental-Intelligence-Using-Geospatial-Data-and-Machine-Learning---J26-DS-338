
"""
J26-DS-338 | Section 04 - Conservation Intelligence
Stage 4.4: Historical Green Persistence (2011-2024)

PRIMARY SOURCE:
Existing annual 100 m grid fractions, filtered to:
    product_type = STABILIZED_ANALYTICAL
    years = 2011 ... 2024

MAIN INDICATORS:
- Historical green-dominant-year proportion
- Historical mean green fraction
- Historical majority-green-year proportion
- Valid historical year count and coverage QA

Green-dominant means LULC class 1 was the dominant
class in a valid year's grid-cell observation.

Majority-green means green_fraction > 0.5.
These are different definitions.

IMPORTANT:
- Stabilized LULC is used only for retrospective analysis.
- No new LULC classification.
- No pressure, SHAP, suitability, AHP or zoning data.
- No Green Importance weights or final score yet.
- Missing historical observations remain missing.
- No arbitrary conservation candidate mask.

OUTPUTS:
CSV, GeoPackage and one QA JSON.
"""

from pathlib import Path
import json

import geopandas as gpd
import numpy as np
import pandas as pd
import pyogrio


ROOT = Path(__file__).resolve().parents[2]

YEARS = list(range(2011, 2025))
N_YEARS = len(YEARS)

ANNUAL_DIR = (
    ROOT
    / "data/processed/urbanization_pressure/"
      "lulc_grid/annual_fractions"
)

CURRENT_CSV = (
    ROOT
    / "data/processed/conservation_intelligence/"
      "current_green/"
      "17b_current_green_condition_2025_100m.csv"
)

CURRENT_GPKG = (
    ROOT
    / "data/processed/conservation_intelligence/"
      "current_green/"
      "17b_current_green_condition_2025_100m.gpkg"
)

CURRENT_QA = (
    ROOT
    / "outputs/reports/conservation_intelligence/"
      "17b_current_green_condition_qa.json"
)

OUTPUT_DIR = (
    ROOT
    / "data/processed/conservation_intelligence/"
      "persistence"
)

REPORT_DIR = (
    ROOT
    / "outputs/reports/conservation_intelligence"
)

CSV_OUT = (
    OUTPUT_DIR
    / "17c_green_persistence_2011_2024_100m.csv"
)

GPKG_OUT = (
    OUTPUT_DIR
    / "17c_green_persistence_2011_2024_100m.gpkg"
)

QA_OUT = (
    REPORT_DIR
    / "17c_green_persistence_qa.json"
)

FRACTION_COLUMNS = [
    "green_fraction",
    "built_fraction",
    "open_bare_fraction",
    "water_fraction",
]

ANNUAL_COLUMNS = [
    "grid_id",
    "year",
    "product_type",
    "kmc_overlap_area_m2",
    "valid_lulc_area_m2",
    "valid_lulc_coverage_fraction",
    "dominant_class_id",
    "dominant_class_fraction",
] + FRACTION_COLUMNS


def require_columns(frame, names, label):
    missing = sorted(
        set(names) - set(frame.columns)
    )

    if missing:
        raise ValueError(
            f"{label}: missing columns {missing}"
        )


def safe_divide(numerator, denominator):
    """Return NaN when the denominator is zero."""

    result = np.full(
        denominator.shape,
        np.nan,
        dtype=float,
    )

    np.divide(
        numerator,
        denominator,
        out=result,
        where=denominator > 0,
    )

    return result


def main():

    print("=" * 72)
    print("SECTION 04 | STAGE 4.4 - HISTORICAL GREEN PERSISTENCE")
    print("=" * 72)

    required_files = [
        CURRENT_CSV,
        CURRENT_GPKG,
        CURRENT_QA,
    ]

    annual_files = {
        year: (
            ANNUAL_DIR
            / f"kaduwela_100m_lulc_fractions_{year}.csv"
        )
        for year in YEARS
    }

    for path in required_files + list(annual_files.values()):
        if not path.is_file():
            raise FileNotFoundError(path)

    previous = json.loads(
        CURRENT_QA.read_text(encoding="utf-8")
    )

    if previous.get("status") != "PASS_CURRENT_GREEN_BASELINE":
        raise ValueError(
            "Stage 4.3 current-green baseline has not passed"
        )

    # ==================================================
    # 1. LOAD CURRENT GRID REFERENCE
    # ==================================================

    print("\n[1] LOAD CURRENT GREEN GRID REFERENCE")

    current = pd.read_csv(CURRENT_CSV)
    current_geo = gpd.read_file(CURRENT_GPKG)

    required_current = [
        "grid_id",
        "grid_row",
        "grid_col",
        "kmc_overlap_area_m2",
        "kmc_overlap_fraction",
        "valid_lulc_coverage_fraction",
        "current_green_fraction",
        "green_condition_status",
        "source_product_type",
    ]

    require_columns(
        current, required_current, "Current green"
    )

    if (
        len(current) != 9240
        or current.grid_id.isna().any()
        or not current.grid_id.is_unique
    ):
        raise ValueError(
            "Current green grid must contain 9240 unique IDs"
        )

    if not current.source_product_type.eq("RAW_RF").all():
        raise ValueError(
            "Current green source must be 2025 RAW_RF"
        )

    if (
        len(current_geo) != 9240
        or current_geo.grid_id.isna().any()
        or not current_geo.grid_id.is_unique
        or set(current_geo.grid_id) != set(current.grid_id)
    ):
        raise ValueError("Current-green GIS IDs mismatch")

    if (
        current_geo.crs is None
        or current_geo.crs.to_epsg() != 32644
        or current_geo.geometry.isna().any()
        or not current_geo.geometry.is_valid.all()
    ):
        raise ValueError("Invalid current-green GIS geometry")

    grid_ids = current["grid_id"].tolist()
    grid_id_set = set(grid_ids)

    reference_area = (
        current["kmc_overlap_area_m2"]
        .to_numpy(dtype=float)
    )

    if (
        not np.isfinite(reference_area).all()
        or (reference_area <= 0).any()
    ):
        raise ValueError("Invalid KMC grid areas")

    print("Current green source: 2025 RAW_RF")
    print("Grid cells:", len(current))
    print("Historical years:", YEARS[0], "to", YEARS[-1])

    # ==================================================
    # 2. READ 14 HISTORICAL STABILIZED PRODUCTS
    # ==================================================

    print("\n[2] LOAD HISTORICAL STABILIZED LULC FRACTIONS")

    green_history = []
    dominant_green_history = []
    majority_green_history = []
    valid_history = []
    coverage_history = []

    product_qa = []

    for year in YEARS:

        annual = pd.read_csv(
            annual_files[year],
            usecols=ANNUAL_COLUMNS,
        )

        product_names = (
            annual["product_type"]
            .astype("string")
            .str.strip()
            .str.upper()
        )

        stabilized = annual.loc[
            product_names.eq("STABILIZED_ANALYTICAL")
        ].copy()

        raw_count = int(
            product_names.eq("RAW_RF").sum()
        )

        if (
            len(stabilized) != 9240
            or stabilized.grid_id.isna().any()
            or not stabilized.grid_id.is_unique
            or set(stabilized.grid_id) != grid_id_set
        ):
            raise ValueError(
                f"{year}: incomplete stabilized grid product"
            )

        if not pd.to_numeric(
            stabilized["year"], errors="coerce"
        ).eq(year).all():
            raise ValueError(
                f"{year}: unexpected historical year values"
            )

        stabilized = (
            stabilized
            .set_index("grid_id")
            .loc[grid_ids]
        )

        stored_area = (
            stabilized["kmc_overlap_area_m2"]
            .to_numpy(dtype=float)
        )

        valid_area = (
            stabilized["valid_lulc_area_m2"]
            .to_numpy(dtype=float)
        )

        coverage = (
            stabilized["valid_lulc_coverage_fraction"]
            .to_numpy(dtype=float)
        )

        fractions = (
            stabilized[FRACTION_COLUMNS]
            .to_numpy(dtype=float)
        )

        dominant_id = pd.to_numeric(
            stabilized["dominant_class_id"],
            errors="coerce",
        ).to_numpy(dtype=float)

        dominant_fraction = pd.to_numeric(
            stabilized["dominant_class_fraction"],
            errors="coerce",
        ).to_numpy(dtype=float)

        valid = valid_area > 1e-9

        # Validate existing annual area-weighted products.
        if (
            not np.isfinite(stored_area).all()
            or not np.isfinite(valid_area).all()
            or not np.isfinite(coverage).all()
            or (valid_area < 0).any()
            or (valid_area > stored_area + 1.0).any()
            or (coverage < -1e-9).any()
            or (coverage > 1 + 1e-9).any()
            or np.max(
                np.abs(stored_area - reference_area)
            ) > 1.0
            or not np.allclose(
                coverage,
                valid_area / stored_area,
                atol=1e-7,
            )
        ):
            raise ValueError(
                f"{year}: area or coverage validation failed"
            )

        if valid.any():

            observed_fractions = fractions[valid]

            if (
                not np.isfinite(
                    observed_fractions
                ).all()
                or (observed_fractions < -1e-8).any()
                or (observed_fractions > 1 + 1e-8).any()
                or not np.allclose(
                    observed_fractions.sum(axis=1),
                    1.0,
                    atol=1e-6,
                )
            ):
                raise ValueError(
                    f"{year}: invalid LULC class fractions"
                )

            # Green = class 1, as established by
            # the original annual LULC fraction generator.
            if not np.isin(
                dominant_id[valid],
                [1, 2, 3, 4],
            ).all():
                raise ValueError(
                    f"{year}: invalid dominant LULC classes"
                )

            if (
                not np.isfinite(
                    dominant_fraction[valid]
                ).all()
                or not np.allclose(
                    dominant_fraction[valid],
                    observed_fractions.max(axis=1),
                    atol=1e-6,
                )
            ):
                raise ValueError(
                    f"{year}: dominant class share mismatch"
                )

        # Missing observations are NaN, not zero green.
        observed_green = np.where(
            valid,
            fractions[:, 0],
            np.nan,
        )

        # Dominant means the largest class share;
        # it does not necessarily mean > 50% green.
        green_dominant = (
            valid & (dominant_id == 1)
        )

        # Alternative historical majority definition.
        majority_green = (
            valid & (fractions[:, 0] > 0.5)
        )

        green_history.append(observed_green)
        dominant_green_history.append(green_dominant)
        majority_green_history.append(majority_green)
        valid_history.append(valid)
        coverage_history.append(coverage)

        product_qa.append({
            "year": year,
            "raw_reference_rows": raw_count,
            "stabilized_rows": len(stabilized),
            "valid_cells": int(valid.sum()),
            "zero_coverage_cells": int((~valid).sum()),
            "mean_coverage": float(coverage.mean()),
        })

        print(
            f"{year} | "
            f"Stabilized: {len(stabilized):,} | "
            f"Valid: {int(valid.sum()):,} | "
            f"Unknown: {int((~valid).sum()):,}"
        )

    # ==================================================
    # 3. CALCULATE HISTORICAL PERSISTENCE INDICATORS
    # ==================================================

    print("\n[3] CALCULATE HISTORICAL PERSISTENCE")

    G = np.stack(green_history)
    D = np.stack(dominant_green_history)
    M = np.stack(majority_green_history)
    V = np.stack(valid_history)
    C = np.stack(coverage_history)

    valid_year_count = V.sum(axis=0).astype(int)

    green_dominant_years = D.sum(axis=0).astype(int)

    majority_green_years = M.sum(axis=0).astype(int)

    # Primary candidate:
    # green-dominant valid years / all valid years.
    dominant_persistence = safe_divide(
        green_dominant_years.astype(float),
        valid_year_count,
    )

    # Continuous historical green-cover alternative.
    mean_green_fraction = safe_divide(
        np.nansum(G, axis=0),
        valid_year_count,
    )

    # A separate majority-green persistence alternative.
    majority_persistence = safe_divide(
        majority_green_years.astype(float),
        valid_year_count,
    )

    minimum_green_fraction = np.min(
        np.where(V, G, np.inf),
        axis=0,
    )

    minimum_green_fraction[
        valid_year_count == 0
    ] = np.nan

    maximum_green_fraction = np.max(
        np.where(V, G, -np.inf),
        axis=0,
    )

    maximum_green_fraction[
        valid_year_count == 0
    ] = np.nan

    mean_coverage_valid = safe_divide(
        np.where(V, C, 0.0).sum(axis=0),
        valid_year_count,
    )

    years_ge80_coverage = (
        V & (C >= 0.80)
    ).sum(axis=0).astype(int)

    years_below80_coverage = (
        V & (C < 0.80)
    ).sum(axis=0).astype(int)

    # Descriptive evidence-quality statuses only.
    history_status = np.full(
        len(current),
        "PARTIAL_VALID_HISTORY",
        dtype=object,
    )

    history_status[
        valid_year_count == N_YEARS
    ] = "COMPLETE_14_VALID_YEARS"

    history_status[
        valid_year_count == 0
    ] = "NO_VALID_HISTORY"

    # The 80% rule is a QA flag, not a deletion
    # threshold or conservation eligibility rule.
    coverage_quality = np.full(
        len(current),
        "HAS_LOW_COVERAGE_HISTORICAL_YEAR",
        dtype=object,
    )

    coverage_quality[
        (valid_year_count == N_YEARS)
        & (years_below80_coverage == 0)
    ] = "ALL_14_YEARS_AT_LEAST_80PCT_COVERAGE"

    coverage_quality[
        valid_year_count == 0
    ] = "NO_VALID_HISTORY"

    print(
        "Cells with all 14 valid years:",
        int((valid_year_count == 14).sum()),
    )

    print(
        "Cells with partial valid history:",
        int(
            (
                (valid_year_count > 0)
                & (valid_year_count < 14)
            ).sum()
        ),
    )

    print(
        "Cells without valid historical years:",
        int((valid_year_count == 0).sum()),
    )

    # ==================================================
    # 4. BUILD GRID-LEVEL PERSISTENCE TABLE
    # ==================================================

    print("\n[4] BUILD PERSISTENCE DATASET")

    result = current[
        [
            "grid_id",
            "grid_row",
            "grid_col",
            "kmc_overlap_area_m2",
            "kmc_overlap_fraction",
            "current_green_fraction",
            "valid_lulc_coverage_fraction",
            "green_condition_status",
        ]
    ].copy()

    result = result.rename(
        columns={
            "valid_lulc_coverage_fraction":
                "current_2025_lulc_coverage_fraction"
        }
    )

    result["historical_start_year"] = 2011
    result["historical_end_year"] = 2024
    result["historical_total_years"] = N_YEARS
    result["historical_product_type"] = (
        "STABILIZED_ANALYTICAL"
    )

    result["historical_valid_year_count"] = (
        valid_year_count
    )

    result["historical_missing_year_count"] = (
        N_YEARS - valid_year_count
    )

    result["historical_green_dominant_years"] = (
        green_dominant_years
    )

    result["historical_green_dominant_year_share"] = (
        dominant_persistence
    )

    result["historical_mean_green_fraction"] = (
        mean_green_fraction
    )

    result["historical_majority_green_years"] = (
        majority_green_years
    )

    result["historical_majority_green_year_share"] = (
        majority_persistence
    )

    result["historical_min_green_fraction"] = (
        minimum_green_fraction
    )

    result["historical_max_green_fraction"] = (
        maximum_green_fraction
    )

    result["historical_mean_valid_coverage"] = (
        mean_coverage_valid
    )

    result["historical_years_ge80pct_coverage"] = (
        years_ge80_coverage
    )

    result["historical_years_below80pct_coverage"] = (
        years_below80_coverage
    )

    result["historical_data_status"] = (
        history_status
    )

    result["historical_coverage_quality"] = (
        coverage_quality
    )

    # ==================================================
    # 5. VALIDATION
    # ==================================================

    print("\n[5] HISTORICAL PERSISTENCE QA")

    checks = {}

    def check(name, condition):
        passed = bool(condition)
        checks[name] = passed
        print(
            f"[{'PASS' if passed else 'FAIL'}] {name}"
        )

    observed_history = valid_year_count > 0

    check(
        "14_historical_years_processed",
        len(YEARS) == 14
        and G.shape == (14, 9240),
    )

    check(
        "all_historical_products_9240",
        len(product_qa) == 14
        and all(
            record["stabilized_rows"] == 9240
            for record in product_qa
        ),
    )

    check(
        "9240_grid_cells_preserved",
        len(result) == 9240
        and result.grid_id.is_unique
        and result.grid_id.notna().all(),
    )

    check(
        "valid_year_counts_0_to_14",
        (
            (valid_year_count >= 0)
            & (valid_year_count <= 14)
        ).all(),
    )

    check(
        "green_dominant_counts_valid",
        (
            green_dominant_years <= valid_year_count
        ).all(),
    )

    check(
        "majority_green_counts_valid",
        (
            majority_green_years <= valid_year_count
        ).all(),
    )

    candidate_arrays = [
        dominant_persistence,
        mean_green_fraction,
        majority_persistence,
        minimum_green_fraction,
        maximum_green_fraction,
    ]

    check(
        "valid_persistence_indicators_0_to_1",
        all(
            np.isfinite(values[observed_history]).all()
            and (
                (values[observed_history] >= -1e-9)
                & (values[observed_history] <= 1 + 1e-9)
            ).all()
            for values in candidate_arrays
        ),
    )

    check(
        "missing_history_remains_nan",
        all(
            np.isnan(values[~observed_history]).all()
            for values in candidate_arrays
        ),
    )

    check(
        "persistence_denominator_reproduced",
        np.allclose(
            dominant_persistence[observed_history],
            green_dominant_years[observed_history]
            / valid_year_count[observed_history],
            atol=1e-12,
        ),
    )

    check(
        "historical_mean_fraction_reproduced",
        np.allclose(
            mean_green_fraction[observed_history],
            np.nansum(
                G, axis=0
            )[observed_history]
            / valid_year_count[observed_history],
            atol=1e-12,
        ),
    )

    check(
        "coverage_quality_counts_valid",
        (
            years_ge80_coverage
            + years_below80_coverage
            == valid_year_count
        ).all(),
    )

    check(
        "current_green_unknowns_preserved",
        result["current_green_fraction"].isna().sum()
        == previous["unknown_green_condition_cells"],
    )

    forbidden = [
        "pressure",
        "shap",
        "suitability",
        "ahp",
        "zoning",
        "priority",
    ]

    check(
        "no_forbidden_variables_in_dataset",
        not any(
            word in column.lower()
            for column in result.columns
            for word in forbidden
        ),
    )

    # Check original clipped GIS areas by grid_id.
    joined_geometry = (
        current_geo
        .set_index("grid_id")
        .loc[result.grid_id]
    )

    geometry_area_error = float(
        np.max(
            np.abs(
                joined_geometry.geometry.area.to_numpy()
                - result[
                    "kmc_overlap_area_m2"
                ].to_numpy(dtype=float)
            )
        )
    )

    check(
        "geometry_area_reconciled",
        geometry_area_error <= 1.0,
    )

    if not all(checks.values()):
        raise SystemExit(
            "Stage 4.4 QA failed. No outputs exported."
        )

    # ==================================================
    # 6. HISTORICAL PERSISTENCE SUMMARY
    # ==================================================

    print("\n[6] HISTORICAL GREEN PERSISTENCE SUMMARY")

    valid_persistence = (
        dominant_persistence[observed_history]
    )

    valid_mean_green = (
        mean_green_fraction[observed_history]
    )

    def distribution(values):
        if len(values) == 0:
            return None

        return {
            "min": float(np.min(values)),
            "p05": float(np.quantile(values, 0.05)),
            "p25": float(np.quantile(values, 0.25)),
            "median": float(np.median(values)),
            "p75": float(np.quantile(values, 0.75)),
            "p95": float(np.quantile(values, 0.95)),
            "max": float(np.max(values)),
        }

    print(
        "Historical green-dominance persistence:",
        distribution(valid_persistence),
    )

    print(
        "Mean historical green fraction:",
        distribution(valid_mean_green),
    )

    # ==================================================
    # 7. EXPORT RESULTS
    # ==================================================

    print("\n[7] EXPORT HISTORICAL PERSISTENCE")

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    result.to_csv(
        CSV_OUT,
        index=False,
    )

    geo = current_geo[
        ["grid_id", "geometry"]
    ].merge(
        result,
        on="grid_id",
        how="inner",
        validate="one_to_one",
    )

    if len(geo) != 9240:
        raise ValueError(
            "GIS join lost historical persistence cells"
        )

    geo.to_file(
        GPKG_OUT,
        layer="historical_green_persistence_100m",
        driver="GPKG",
    )

    saved_csv = pd.read_csv(CSV_OUT)
    saved_gis_info = pyogrio.read_info(GPKG_OUT)

    check(
        "csv_export_9240",
        len(saved_csv) == 9240
        and saved_csv.grid_id.is_unique,
    )

    check(
        "gis_export_9240",
        saved_gis_info["features"] == 9240,
    )

    check(
        "missing_values_preserved_in_export",
        int(
            saved_csv[
                "historical_green_dominant_year_share"
            ].isna().sum()
        ) == int((valid_year_count == 0).sum()),
    )

    status = (
        "PASS_HISTORICAL_PERSISTENCE"
        if all(checks.values())
        else "REVIEW_REQUIRED"
    )

    qa = {
        "section": "04_conservation_intelligence",
        "stage": "4.4",
        "status": status,
        "period": "2011-2024",
        "years": YEARS,
        "historical_product": "STABILIZED_ANALYTICAL",
        "grid_cells": len(result),
        "cells_with_14_valid_years": int(
            (valid_year_count == 14).sum()
        ),
        "cells_with_partial_valid_history": int(
            (
                (valid_year_count > 0)
                & (valid_year_count < 14)
            ).sum()
        ),
        "cells_without_valid_history": int(
            (valid_year_count == 0).sum()
        ),
        "dominant_green_persistence_distribution": (
            distribution(valid_persistence)
        ),
        "historical_mean_green_fraction_distribution": (
            distribution(valid_mean_green)
        ),
        "annual_source_qa": product_qa,
        "max_geometry_area_error_m2": (
            geometry_area_error
        ),
        "checks": checks,
        "primary_candidate_definition": (
            "Number of valid years with dominant LULC "
            "class 1 (Green), divided by valid historical "
            "year count"
        ),
        "alternative_indicator_definitions": {
            "historical_mean_green_fraction": (
                "Mean green fraction across valid "
                "historical observations"
            ),
            "historical_majority_green_year_share": (
                "Share of valid years with green fraction > 0.5"
            ),
        },
        "methodology_notes": [
            "Historical stabilized fractions are for "
            "retrospective analysis only.",
            "Centered stabilization may incorporate "
            "adjacent later years; it is not a causal predictor.",
            "Zero-coverage historical observations are "
            "excluded from the denominator, not set to zero.",
            "Valid-year counts and coverage quality are retained.",
            "The 80% coverage flag is diagnostic only.",
            "No final persistence indicator selected yet.",
            "No Green Importance score calculated yet.",
            "No pressure, SHAP, suitability, AHP or zoning used.",
        ],
    }

    QA_OUT.write_text(
        json.dumps(qa, indent=2),
        encoding="utf-8",
    )

    print("CSV:", CSV_OUT)
    print("GPKG:", GPKG_OUT)
    print("QA:", QA_OUT)

    print("\n" + "=" * 72)
    print("STAGE 4.4:", status)
    print("Historical years:", N_YEARS)
    print("Grid cells:", len(result))
    print("Green Importance score: NOT CALCULATED")
    print("Final persistence criterion: NOT YET SELECTED")
    print("Future pressure used: NO")
    print("=" * 72)

    if status != "PASS_HISTORICAL_PERSISTENCE":
        raise SystemExit(
            "Historical persistence export QA failed."
        )


if __name__ == "__main__":
    main()
