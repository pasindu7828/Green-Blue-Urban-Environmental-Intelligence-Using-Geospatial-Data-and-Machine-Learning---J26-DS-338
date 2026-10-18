
"""
J26-DS-338 | Section 04 - Conservation Intelligence
Stages 4.1-4.2: Conservation Framework and Data Readiness

CORRECTED VERSION:
The 2025 annual fractions CSV contains two distinct products:
  RAW_RF                - Primary current green baseline
  STABILIZED_ANALYTICAL - Retrospective analytical reference

Validate both products separately.
Use RAW_RF only for current-green-condition checks.

No historical datasets or ML models are modified.
No Green Importance or Conservation Priority scores are calculated.
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio


ROOT = Path(__file__).resolve().parents[2]

REPORT = ROOT / "outputs/reports/conservation_intelligence"

P = {
    "grid":
        "data/processed/urbanization_pressure/grid/"
        "kaduwela_100m_modelling_grid.gpkg",

    "boundary":
        "data/processed/boundaries/kmc/"
        "kaduwela_kmc_boundary_utm44n.gpkg",

    "fractions_2025":
        "data/processed/urbanization_pressure/lulc_grid/"
        "annual_fractions/kaduwela_100m_lulc_fractions_2025.csv",

    "raw_2025":
        "data/processed/lulc/annual_maps/raw/"
        "Kaduwela_LULC_RAW_2025.tif",

    "robust_transition":
        "data/processed/lulc/transitions/"
        "Kaduwela_Green_Transition_2012_2014_to_2021_2023_ROBUST.tif",

    "strict_transition":
        "data/processed/lulc/transitions/"
        "Kaduwela_Green_Transition_2012_2014_to_2021_2023_STRICT.tif",

    "eligibility":
        "data/processed/urbanization_pressure/inference/"
        "14b_2025_inference_eligibility.csv",

    "pressure":
        "data/processed/urbanization_pressure/inference/"
        "14c_2026_2028_urbanization_pressure_scores.csv",

    "fraction_generator":
        "src/02_urbanization_pressure/"
        "11b_annual_lulc_to_100m_grid.py",
}


STABILIZED = [
    ROOT / (
        "data/processed/lulc/annual_maps/stabilized_mode_3yr/"
        f"Kaduwela_LULC_STABILIZED_{year}.tif"
    )
    for year in range(2011, 2025)
]


REGISTRY = [
    (
        "100 m KMC grid",
        P["grid"],
        "CORE",
        "Primary spatial analysis unit",
    ),
    (
        "KMC boundary",
        P["boundary"],
        "CORE",
        "Official study area",
    ),
    (
        "2025 RAW_RF grid fractions",
        P["fractions_2025"],
        "CORE",
        "Primary current green condition; filter RAW_RF",
    ),
    (
        "2025 RAW LULC raster",
        P["raw_2025"],
        "CORE",
        "Primary current-year land-cover source",
    ),
    (
        "Stabilized LULC 2011-2024",
        "data/processed/lulc/annual_maps/stabilized_mode_3yr/",
        "CORE",
        "Retrospective historical green persistence",
    ),
    (
        "Robust historical Green-to-Built transition",
        P["robust_transition"],
        "HISTORICAL_THREAT",
        "Historical loss exposure only",
    ),
    (
        "Strict historical Green-to-Built transition",
        P["strict_transition"],
        "SENSITIVITY",
        "Alternative historical loss definition",
    ),
    (
        "Frozen future pressure eligibility",
        P["eligibility"],
        "FUTURE_THREAT_ONLY",
        "Differentiate SCORED and NOT_IN_DOMAIN",
    ),
    (
        "Frozen future pressure scores",
        P["pressure"],
        "FUTURE_THREAT_ONLY",
        "Introduce after Green Importance is frozen",
    ),
    (
        "2026 RAW LULC",
        "data/processed/lulc/annual_maps/raw/"
        "Kaduwela_LULC_RAW_2026.tif",
        "SENSITIVITY",
        "Provisional-year comparison",
    ),
    (
        "ESA WorldCover 2021",
        "data/processed/reference_lulc/",
        "REFERENCE",
        "Not 2025 ground truth",
    ),
    (
        "Provisional UDA zoning",
        "",
        "CONTEXT_ONLY",
        "Not used as an authoritative restriction",
    ),
    (
        "Development Suitability and AHP",
        "data/processed/development_suitability/",
        "EXCLUDED",
        "Not an input to Green Importance or Conservation Priority",
    ),
    (
        "Pressure and SHAP",
        "data/processed/urbanization_pressure/",
        "EXCLUDED_FROM_GREEN_IMPORTANCE",
        "Pressure used only later as future threat; SHAP excluded",
    ),
]


def booleans(series):
    return (
        series.astype("string")
        .str.strip()
        .str.lower()
        .isin(["true", "1", "yes"])
        .to_numpy(dtype=bool)
    )


def read_csv(path, label):
    df = pd.read_csv(path)

    if "grid_id" not in df.columns:
        raise ValueError(f"{label}: grid_id column missing")

    return df


def raster_info(path):

    with rasterio.open(path) as src:

        array = src.read(1, masked=True)

        values, counts = np.unique(
            array.compressed(),
            return_counts=True,
        )

        histogram = {
            str(int(value)): int(count)
            for value, count in zip(values, counts)
        }

        return {
            "crs": (
                src.crs.to_epsg()
                if src.crs is not None
                else None
            ),
            "shape": [
                src.height,
                src.width,
            ],
            "resolution_m": [
                float(abs(src.res[0])),
                float(abs(src.res[1])),
            ],
            "transform": tuple(src.transform),
            "nodata": src.nodata,
            "histogram": histogram,
        }


def main():

    print("=" * 75)
    print(
        "SECTION 04 | STAGES 4.1-4.2 "
        "- CONSERVATION DATA READINESS"
    )
    print("=" * 75)

    required = {
        key: ROOT / relative_path
        for key, relative_path in P.items()
    }

    for path in [
        *required.values(),
        *STABILIZED,
    ]:
        if not path.is_file():
            raise FileNotFoundError(
                f"Required input missing: {path}"
            )

    checks = {}
    observations = {}

    def check(name, condition):

        passed = bool(condition)
        checks[name] = passed

        print(
            f"[{'PASS' if passed else 'FAIL'}] {name}"
        )

    # ==================================================
    # 1. GRID AND STUDY AREA
    # ==================================================

    print("\n[1] GRID / STUDY AREA")

    grid = gpd.read_file(required["grid"])
    kmc = gpd.read_file(required["boundary"])

    grid_ids = set(grid["grid_id"].tolist())

    kmc_area = float(
        kmc.geometry.area.sum()
    )

    kmc_geometry = kmc.geometry.union_all()

    # The modelling grid may contain full 100 m polygons.
    # Use Grid intersection KMC for actual analysis area.
    actual_overlap = (
        grid.geometry
        .intersection(kmc_geometry)
        .area
        .to_numpy(dtype=float)
    )

    stored_overlap = (
        grid["kmc_overlap_area_m2"]
        .to_numpy(dtype=float)
    )

    check(
        "grid_9240_unique_nonnull",
        len(grid) == 9240
        and grid.grid_id.notna().all()
        and grid.grid_id.is_unique,
    )

    check(
        "grid_and_boundary_epsg32644",
        grid.crs is not None
        and kmc.crs is not None
        and grid.crs.to_epsg() == 32644
        and kmc.crs.to_epsg() == 32644,
    )

    check(
        "grid_geometries_valid",
        grid.geometry.notna().all()
        and grid.geometry.is_valid.all(),
    )

    check(
        "kmc_overlap_area_matches_geometry",
        np.isfinite(stored_overlap).all()
        and (stored_overlap > 0).all()
        and np.max(
            np.abs(actual_overlap - stored_overlap)
        ) <= 1.0,
    )

    check(
        "grid_area_matches_kmc",
        abs(
            float(actual_overlap.sum()) - kmc_area
        ) <= 1.0,
    )

    observations["study_area_km2"] = (
        kmc_area / 1e6
    )

    print(
        f"Cells: {len(grid):,} | "
        f"KMC area: {kmc_area / 1e6:.6f} km2"
    )

    # ==================================================
    # 2. TWO-PRODUCT 2025 LULC SOURCE
    # ==================================================

    print("\n[2] 2025 LULC FRACTIONS / PROVENANCE")

    all_fractions = read_csv(
        required["fractions_2025"],
        "2025 fractions",
    )

    columns = [
        "grid_id",
        "year",
        "product_type",
        "future_informed",
        "provisional_year",
        "kmc_overlap_area_m2",
        "valid_lulc_area_m2",
        "valid_lulc_coverage_fraction",
        "green_area_m2",
        "built_area_m2",
        "open_bare_area_m2",
        "water_area_m2",
        "green_fraction",
        "built_fraction",
        "open_bare_fraction",
        "water_fraction",
    ]

    missing = sorted(
        set(columns) - set(all_fractions.columns)
    )

    if missing:
        raise ValueError(
            f"Missing fraction columns: {missing}"
        )

    products = (
        all_fractions["product_type"]
        .astype("string")
        .str.strip()
        .str.upper()
    )

    product_counts = {
        str(key): int(value)
        for key, value in products.value_counts(
            dropna=False
        ).items()
    }

    print("Product types:", product_counts)

    raw = all_fractions.loc[
        products.eq("RAW_RF")
    ].copy()

    stabilized = all_fractions.loc[
        products.eq("STABILIZED_ANALYTICAL")
    ].copy()

    def complete_product(df):
        return (
            len(df) == 9240
            and df.grid_id.notna().all()
            and df.grid_id.is_unique
            and set(df.grid_id) == grid_ids
        )

    check(
        "2025_two_product_structure_valid",
        len(all_fractions) == 18480
        and set(product_counts) == {
            "RAW_RF",
            "STABILIZED_ANALYTICAL",
        }
        and complete_product(stabilized),
    )

    check(
        "2025_fraction_grid_ids_match",
        complete_product(raw),
    )

    check(
        "2025_product_tag_is_RAW",
        len(raw) == 9240
        and raw["product_type"]
        .astype("string")
        .str.strip()
        .str.upper()
        .eq("RAW_RF")
        .all(),
    )

    print("Primary RAW_RF rows:", len(raw))
    print(
        "Stabilized reference rows:",
        len(stabilized),
    )

    # Critical: use RAW_RF for all remaining current-green
    # checks, not the stacked two-product table.
    frac = raw

    check(
        "2025_year_correct",
        pd.to_numeric(
            frac["year"],
            errors="coerce",
        ).eq(2025).all(),
    )

    check(
        "2025_not_future_informed",
        not booleans(
            frac["future_informed"]
        ).any(),
    )

    check(
        "2025_not_provisional",
        not booleans(
            frac["provisional_year"]
        ).any(),
    )

    # ==================================================
    # 3. CURRENT GREEN COVERAGE / FRACTION VALIDATION
    # ==================================================

    print("\n[3] RAW 2025 COVERAGE VALIDATION")

    valid_area = (
        frac["valid_lulc_area_m2"]
        .to_numpy(dtype=float)
    )

    total_area = (
        frac["kmc_overlap_area_m2"]
        .to_numpy(dtype=float)
    )

    coverage = (
        frac["valid_lulc_coverage_fraction"]
        .to_numpy(dtype=float)
    )

    class_areas = frac[
        [
            "green_area_m2",
            "built_area_m2",
            "open_bare_area_m2",
            "water_area_m2",
        ]
    ].to_numpy(dtype=float)

    class_fractions = frac[
        [
            "green_fraction",
            "built_fraction",
            "open_bare_fraction",
            "water_fraction",
        ]
    ].to_numpy(dtype=float)

    has_valid_lulc = valid_area > 1e-9

    check(
        "2025_area_and_coverage_valid",
        np.isfinite(valid_area).all()
        and np.isfinite(total_area).all()
        and np.isfinite(coverage).all()
        and (valid_area >= 0).all()
        and (total_area > 0).all()
        and (valid_area <= total_area + 1.0).all()
        and (coverage >= -1e-9).all()
        and (coverage <= 1 + 1e-9).all(),
    )

    check(
        "2025_coverage_denominator_valid",
        np.allclose(
            coverage,
            valid_area / total_area,
            atol=1e-7,
        ),
    )

    check(
        "2025_class_areas_sum_to_valid_area",
        np.isfinite(class_areas).all()
        and (class_areas >= -1e-7).all()
        and np.allclose(
            class_areas.sum(axis=1),
            valid_area,
            atol=1.0,
        ),
    )

    check(
        "2025_fractions_match_valid_area",
        np.isfinite(
            class_fractions[has_valid_lulc]
        ).all()
        and np.allclose(
            class_fractions[has_valid_lulc],
            class_areas[has_valid_lulc]
            / valid_area[has_valid_lulc, None],
            atol=1e-6,
        ),
    )

    check(
        "2025_valid_fractions_sum_to_one",
        np.isfinite(
            class_fractions[has_valid_lulc]
        ).all()
        and (
            class_fractions[has_valid_lulc] >= -1e-8
        ).all()
        and (
            class_fractions[has_valid_lulc] <= 1 + 1e-8
        ).all()
        and np.allclose(
            class_fractions[has_valid_lulc].sum(axis=1),
            1.0,
            atol=1e-6,
        ),
    )

    area_lookup = grid.set_index(
        "grid_id"
    )["kmc_overlap_area_m2"]

    matched_area = (
        frac["grid_id"]
        .map(area_lookup)
        .to_numpy(dtype=float)
    )

    check(
        "2025_kmc_areas_match_grid",
        np.isfinite(matched_area).all()
        and np.max(
            np.abs(total_area - matched_area)
        ) <= 1.0,
    )

    observations["fraction_product_types"] = (
        product_counts
    )

    observations["primary_green_source"] = (
        "2025 RAW_RF records only"
    )

    observations["raw_baseline_cells"] = len(raw)

    observations["stabilized_reference_cells"] = (
        len(stabilized)
    )

    observations["current_fraction_denominator"] = (
        "VALID_CLASSIFIED_LULC_AREA"
    )

    observations["positive_valid_coverage_cells"] = (
        int(has_valid_lulc.sum())
    )

    observations["zero_valid_coverage_cells"] = (
        int((~has_valid_lulc).sum())
    )

    observations["coverage_percentiles"] = {
        f"p{p:02d}": float(
            np.percentile(coverage, p)
        )
        for p in [0, 5, 25, 50, 75, 95, 100]
    }

    observations["cells_below_50pct_valid_coverage"] = (
        int((coverage < 0.5).sum())
    )

    observations["cells_below_80pct_valid_coverage"] = (
        int((coverage < 0.8).sum())
    )

    observations["zero_coverage_handling"] = (
        "UNKNOWN_GREEN_CONDITION, never assumed zero green"
    )

    print(
        "Positive-coverage cells:",
        observations["positive_valid_coverage_cells"],
    )

    print(
        "Zero-coverage cells:",
        observations["zero_valid_coverage_cells"],
    )

    print(
        "Coverage percentiles:",
        observations["coverage_percentiles"],
    )

    print(
        "Zero-coverage green condition must remain UNKNOWN."
    )

    # ==================================================
    # 4. HISTORICAL RASTER READINESS
    # ==================================================

    print("\n[4] HISTORICAL LULC RASTER READINESS")

    raw_raster = raster_info(
        required["raw_2025"]
    )

    history = [
        raster_info(path)
        for path in STABILIZED
    ]

    transitions = {
        "ROBUST": raster_info(
            required["robust_transition"]
        ),
        "STRICT": raster_info(
            required["strict_transition"]
        ),
    }

    def aligned(meta):
        return (
            meta["crs"] == raw_raster["crs"] == 32644
            and meta["shape"] == raw_raster["shape"]
            and np.allclose(
                meta["transform"],
                raw_raster["transform"],
                atol=1e-8,
            )
            and np.allclose(
                meta["resolution_m"],
                [30.0, 30.0],
            )
        )

    check(
        "2025_raw_raster_has_expected_classes",
        aligned(raw_raster)
        and set(raw_raster["histogram"]).issubset(
            {"1", "2", "3", "4"}
        )
        and "1" in raw_raster["histogram"],
    )

    check(
        "historical_14_years_present_aligned",
        len(history) == 14
        and all(aligned(meta) for meta in history),
    )

    check(
        "transition_rasters_aligned",
        all(
            aligned(meta)
            for meta in transitions.values()
        ),
    )

    robust_area = (
        transitions["ROBUST"]["histogram"].get(
            "2", 0
        ) * 900.0 / 1e6
    )

    strict_area = (
        transitions["STRICT"]["histogram"].get(
            "2", 0
        ) * 900.0 / 1e6
    )

    print(
        f"Transition code 2: "
        f"ROBUST={robust_area:.4f} km2, "
        f"STRICT={strict_area:.4f} km2"
    )

    check(
        "transition_code2_matches_frozen_green_to_built_areas",
        np.isclose(
            robust_area, 7.9821, atol=1e-5
        )
        and np.isclose(
            strict_area, 4.8267, atol=1e-5
        ),
    )

    observations["historical_period"] = (
        "2011-2024, 14 retrospective stabilized years"
    )

    observations["raw_raster_histogram"] = (
        raw_raster["histogram"]
    )

    observations["transition_code2_candidate"] = {
        "robust_km2": robust_area,
        "strict_km2": strict_area,
        "codebook_verified": False,
    }

    # ==================================================
    # 5. FUTURE PRESSURE DOMAIN READINESS ONLY
    # ==================================================

    print("\n[5] FUTURE PRESSURE - DOMAIN AUDIT ONLY")

    eligibility = read_csv(
        required["eligibility"],
        "Pressure eligibility",
    )

    pressure = read_csv(
        required["pressure"],
        "Pressure scores",
    )

    for name, df in [
        ("eligibility", eligibility),
        ("pressure", pressure),
    ]:
        if (
            df.grid_id.isna().any()
            or not df.grid_id.is_unique
        ):
            raise ValueError(
                f"Invalid grid IDs: {name}"
            )

    if (
        "final_inference_ready" not in eligibility
        or "urbanization_pressure_score" not in pressure
    ):
        raise ValueError(
            "Pressure eligibility or score column missing"
        )

    ready_ids = set(
        eligibility.loc[
            booleans(
                eligibility["final_inference_ready"]
            ),
            "grid_id",
        ]
    )

    scored_ids = set(
        pressure["grid_id"]
    )

    check(
        "pressure_eligibility_all_grid_cells",
        len(eligibility) == 9240
        and set(eligibility.grid_id) == grid_ids,
    )

    check(
        "pressure_scores_match_ready_domain",
        scored_ids == ready_ids
        and scored_ids.issubset(grid_ids),
    )

    check(
        "pressure_frozen_origin_2025",
        pd.to_numeric(
            pressure["origin_year"],
            errors="coerce",
        ).eq(2025).all(),
    )

    pressure_scores = pd.to_numeric(
        pressure["urbanization_pressure_score"],
        errors="coerce",
    ).to_numpy(dtype=float)

    check(
        "pressure_scores_finite_0_to_1",
        np.isfinite(pressure_scores).all()
        and (pressure_scores >= 0).all()
        and (pressure_scores <= 1).all(),
    )

    observations["pressure_scored_cells"] = (
        len(scored_ids)
    )

    observations["pressure_NOT_IN_DOMAIN_cells"] = (
        len(grid_ids - scored_ids)
    )

    observations["pressure_usage_gate"] = (
        "Future threat only, after Green Importance freeze"
    )

    print(
        "Scored cells:",
        len(scored_ids),
    )

    print(
        "NOT_IN_DOMAIN cells:",
        len(grid_ids - scored_ids),
    )

    print(
        "No future pressure added to Green Importance."
    )

    # ==================================================
    # 6. OUTPUT REPORTS
    # ==================================================

    print("\n[6] RESULT AND EVIDENCE")

    passed_count = sum(checks.values())

    total_count = len(checks)

    for name, passed in checks.items():

        print(
            f"[{'PASS' if passed else 'FAIL'}] {name}"
        )

    status = (
        "PASS_SOURCE_READINESS_PENDING_CODEBOOK"
        if all(checks.values())
        else "REVIEW_REQUIRED"
    )

    report = {
        "section": "04_conservation_intelligence",
        "stage": "4.1-4.2",
        "status": status,
        "passed_checks": passed_count,
        "total_checks": total_count,
        "checks": checks,
        "observations": observations,
        "unresolved_before_indicator_calculation": [
            "Verify RAW_RF class meaning and provenance "
            "against the original generator script.",
            "Confirm historical transition codebook.",
            "Treat zero-valid-coverage cells as unknown.",
            "Justify Green Importance aggregation method "
            "before calculating final scores.",
        ],
        "methodological_separation": [
            "Pressure excluded from Green Importance.",
            "Development Suitability excluded.",
            "SHAP excluded.",
            "Provisional zoning excluded.",
            "Historical loss belongs to threat domain.",
            "NOT_IN_DOMAIN is not zero pressure.",
        ],
    }

    REPORT.mkdir(
        parents=True,
        exist_ok=True,
    )

    registry_path = (
        REPORT / "17a_data_registry.csv"
    )

    qa_path = (
        REPORT / "17a_data_readiness_qa.json"
    )

    pd.DataFrame(
        REGISTRY,
        columns=[
            "dataset",
            "relative_path",
            "role",
            "reason",
        ],
    ).to_csv(
        registry_path,
        index=False,
    )

    qa_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print("\nRegistry:", registry_path)
    print("QA report:", qa_path)

    print(
        f"\nSTAGES 4.1-4.2: "
        f"{status} ({passed_count}/{total_count} checks)"
    )

    if not all(checks.values()):
        raise SystemExit(
            "Data-readiness review required."
        )


if __name__ == "__main__":
    main()
