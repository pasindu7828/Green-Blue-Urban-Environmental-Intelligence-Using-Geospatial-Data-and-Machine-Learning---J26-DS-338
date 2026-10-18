
"""
SECTION 03 - DEVELOPMENT SUITABILITY
Stages 3.1 and 3.2: Criteria Framework and Data Readiness

Research: J26-DS-338
Study area: Kaduwela Municipal Council

Purpose:
- Register candidate GIS-MCDA criteria.
- Audit essential geospatial inputs.
- Confirm 100 m grid and KMC boundary consistency.
- Document methodological exclusions.

No AHP weights, standardization or suitability scores.
"""

from pathlib import Path
import csv
import json

import geopandas as gpd
import numpy as np
import pandas as pd
import pyogrio
import rasterio
from shapely.geometry import box


ROOT = Path(__file__).resolve().parents[2]

REPORT_DIR = (
    ROOT / "outputs" / "reports" / "development_suitability"
)

SOURCES = {
    "grid": (
        "data/processed/urbanization_pressure/grid/"
        "kaduwela_100m_modelling_grid.gpkg"
    ),
    "boundary": (
        "data/processed/boundaries/kmc/"
        "kaduwela_kmc_boundary_utm44n.gpkg"
    ),
    "roads": "data/processed/osm/kaduwela_roads_2026.gpkg",
    "buildings": (
        "data/processed/osm/kaduwela_buildings_2026.gpkg"
    ),
    "gedtm": "data/processed/dem/gedtm30_kmc_30m.tif",
    "nasadem": "data/processed/dem/nasadem_kmc_30m.tif",
    "worldcover": (
        "data/processed/reference_lulc/"
        "esa_worldcover_2021_kmc_10m.tif"
    ),
}

CRITERIA = [
    {
        "id": "C1",
        "criterion": "Higher-order road accessibility",
        "source": "OSM roads 2026",
        "raw_measure": "Distance to selected higher-order roads (m)",
        "role": "CANDIDATE_WEIGHTED",
        "direction": "COST_PROVISIONAL",
        "decision": "Confirm accessibility scoring in Stage 3.5",
    },
    {
        "id": "C2",
        "criterion": "Local road-network context",
        "source": "OSM roads 2026",
        "raw_measure": "Road density at 250, 500 and 1000 m",
        "role": "CANDIDATE_WEIGHTED",
        "direction": "BENEFIT_PROVISIONAL",
        "decision": "Select one radius in Stage 3.4",
    },
    {
        "id": "C3",
        "criterion": "Existing development context",
        "source": "OSM building footprints 2026",
        "raw_measure": (
            "Cell occupation and surrounding building "
            "footprint context at 250 and 500 m"
        ),
        "role": "CANDIDATE_WEIGHTED",
        "direction": "REVIEW_REQUIRED",
        "decision": (
            "Avoid double counting and rewarding "
            "already saturated cells"
        ),
    },
    {
        "id": "C4",
        "criterion": "Terrain slope",
        "source": "GEDTM30",
        "raw_measure": "Slope in degrees",
        "role": "CANDIDATE_WEIGHTED",
        "direction": "COST_PROVISIONAL",
        "decision": "Justify scoring thresholds in Stage 3.5",
    },
    {
        "id": "C5",
        "criterion": "Elevation",
        "source": "GEDTM30",
        "raw_measure": "Elevation above vertical datum (m)",
        "role": "CANDIDATE_REVIEW",
        "direction": "UNDECIDED",
        "decision": (
            "Retain only with defensible planning "
            "meaning and direction"
        ),
    },
    {
        "id": "C6",
        "criterion": "Land-cover screening",
        "source": "ESA WorldCover 2021",
        "raw_measure": "Reference land-cover class",
        "role": "REFERENCE_ONLY",
        "direction": "NOT_APPLICABLE",
        "decision": (
            "Not a weighted criterion without "
            "separate justification"
        ),
    },
]


def main():
    print("=" * 65)
    print("SECTION 03 - STAGES 3.1 AND 3.2")
    print("CRITERIA FRAMEWORK AND DATA READINESS")
    print("=" * 65)

    checks = []

    def check(name, passed, observed):
        result = {
            "check": name,
            "passed": bool(passed),
            "observed": str(observed),
        }
        checks.append(result)

        label = "PASS" if passed else "FAIL"
        print(f"[{label}] {name}: {observed}")

    # ---------------------------------------------------
    # 1. Verify input files
    # ---------------------------------------------------

    print("\n[1] SOURCE FILE CHECK")

    paths = {
        name: ROOT / relative
        for name, relative in SOURCES.items()
    }

    for name, path in paths.items():
        check(
            f"Source available: {name}",
            path.is_file(),
            path.relative_to(ROOT),
        )

    missing = [
        name for name, path in paths.items()
        if not path.is_file()
    ]

    if missing:
        raise FileNotFoundError(
            f"Missing required datasets: {missing}"
        )

    # ---------------------------------------------------
    # 2. Inspect grid and official KMC boundary
    # ---------------------------------------------------

    print("\n[2] GRID AND BOUNDARY VALIDATION")

    grid = gpd.read_file(paths["grid"])
    kmc = gpd.read_file(paths["boundary"])

    check("Grid count", len(grid) == 9240, len(grid))

    check(
        "Unique grid IDs",
        grid["grid_id"].notna().all()
        and grid["grid_id"].is_unique,
        grid["grid_id"].nunique(),
    )

    check(
        "Grid CRS",
        grid.crs is not None
        and grid.crs.to_epsg() == 32644,
        grid.crs,
    )

    check(
        "Boundary CRS",
        kmc.crs is not None
        and kmc.crs.to_epsg() == 32644,
        kmc.crs,
    )

    check(
        "Single KMC boundary",
        len(kmc) == 1,
        len(kmc),
    )

    check(
        "Grid geometries valid",
        grid.geometry.notna().all()
        and grid.geometry.is_valid.all(),
        "All valid" if grid.geometry.is_valid.all()
        else "Invalid geometries detected",
    )

    check(
        "KMC geometry valid",
        kmc.geometry.notna().all()
        and kmc.geometry.is_valid.all(),
        "Valid" if kmc.geometry.is_valid.all()
        else "Invalid",
    )

    overlap_area = pd.to_numeric(
        grid["kmc_overlap_area_m2"],
        errors="coerce",
    )

    check(
        "Positive KMC overlap areas",
        overlap_area.notna().all()
        and (overlap_area > 0).all(),
        f"{int((overlap_area > 0).sum())} positive cells",
    )

    check(
        "Overlap not greater than grid geometry",
        (
            overlap_area <= grid.geometry.area + 1.0
        ).all(),
        "Area tolerance: 1 m2",
    )

    kmc_area = float(kmc.geometry.area.sum())
    grid_overlap_total = float(overlap_area.sum())

    area_difference = abs(kmc_area - grid_overlap_total)

    check(
        "KMC/grid total area reconciliation",
        area_difference <= 1.0,
        f"Difference = {area_difference:.6f} m2",
    )

    print(f"KMC area: {kmc_area / 1e6:.6f} km2")
    print(
        "Summed grid/KMC overlap: "
        f"{grid_overlap_total / 1e6:.6f} km2"
    )

    # ---------------------------------------------------
    # 3. Vector source schema checks
    # ---------------------------------------------------

    print("\n[3] OSM SOURCE VALIDATION")

    vector_specs = {
        "roads": {
            "count": 6543,
            "fields": {"highway_clean", "length_m"},
            "geometry": "line",
        },
        "buildings": {
            "count": 33673,
            "fields": {"building_clean", "area_m2"},
            "geometry": "polygon",
        },
    }

    for name, spec in vector_specs.items():
        info = pyogrio.read_info(paths[name])

        fields = set(info["fields"])
        geom_type = str(info["geometry_type"]).lower()

        crs = rasterio.crs.CRS.from_user_input(
            info["crs"]
        )

        check(
            f"{name} feature count",
            info["features"] == spec["count"],
            info["features"],
        )

        check(
            f"{name} CRS",
            crs.to_epsg() == 32644,
            crs,
        )

        check(
            f"{name} required attributes",
            spec["fields"].issubset(fields),
            sorted(spec["fields"] - fields)
            if not spec["fields"].issubset(fields)
            else "All present",
        )

        check(
            f"{name} geometry type",
            spec["geometry"] in geom_type,
            info["geometry_type"],
        )

    # ---------------------------------------------------
    # 4. Raster source checks
    # ---------------------------------------------------

    print("\n[4] RASTER SOURCE VALIDATION")

    kmc_geometry = kmc.geometry.iloc[0]

    for name in ["gedtm", "nasadem", "worldcover"]:
        with rasterio.open(paths[name]) as ds:
            values = ds.read(1, masked=True)

            valid_count = int(
                np.count_nonzero(
                    ~np.ma.getmaskarray(values)
                    & np.isfinite(
                        np.ma.getdata(values)
                    )
                )
            )

            raster_extent = box(*ds.bounds)

            check(
                f"{name} CRS",
                ds.crs is not None
                and ds.crs.to_epsg() == 32644,
                ds.crs,
            )

            check(
                f"{name} overlaps KMC",
                raster_extent.intersects(kmc_geometry),
                ds.bounds,
            )

            check(
                f"{name} contains valid pixels",
                valid_count > 0,
                f"{valid_count} valid pixels",
            )

            check(
                f"{name} single-band source",
                ds.count == 1,
                f"{ds.count} band(s)",
            )

    # ---------------------------------------------------
    # 5. Save compact research evidence
    # ---------------------------------------------------

    print("\n[5] SAVE RESEARCH EVIDENCE")

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    criteria_path = REPORT_DIR / "03_criteria_registry.csv"

    with criteria_path.open(
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(CRITERIA[0].keys()),
        )
        writer.writeheader()
        writer.writerows(CRITERIA)

    passed_count = sum(
        item["passed"] for item in checks
    )

    status = (
        "PASS" if passed_count == len(checks)
        else "FAIL"
    )

    report = {
        "research_component": (
            "Urban Green Transition, Urbanization "
            "Pressure and Conservation Intelligence"
        ),
        "stages": ["3.1", "3.2"],
        "status": status,
        "checks_passed": passed_count,
        "checks_total": len(checks),
        "grid_cells": len(grid),
        "kmc_area_km2": round(kmc_area / 1e6, 6),
        "candidate_criteria_count": len(CRITERIA),
        "primary_terrain_source": "GEDTM30",
        "secondary_terrain_source": (
            "NASADEM - sensitivity/reference only"
        ),
        "planning_zoning": (
            "Provisional UDA data - context only"
        ),
        "excluded_core_inputs": [
            "Urbanization Pressure",
            "SHAP explanations",
            "Conservation Priority",
            "Future pressure predictions",
            "Provisional UDA zoning",
        ],
        "methodology_controls": {
            "ahp_weights_assigned": False,
            "criteria_standardized": False,
            "suitability_score_calculated": False,
            "criterion_directions_finalized": False,
        },
        "limitations": [
            "Input readiness is not suitability accuracy.",
            "Detailed per-cell raster coverage is not "
            "yet validated.",
            "Previous 39 missing terrain cells must "
            "be investigated in Stage 3.3.",
            "OSM completeness is not independently "
            "validated by this audit.",
            "Final criteria require redundancy and "
            "planning-justification review.",
        ],
        "checks": checks,
    }

    qa_path = REPORT_DIR / "03_data_readiness_qa.json"

    with qa_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"\nCriteria registry: {criteria_path}")
    print(f"Data readiness QA: {qa_path}")

    print("\n" + "=" * 65)
    print(f"STAGES 3.1 AND 3.2: {status}")
    print(f"QA checks: {passed_count}/{len(checks)}")
    print("Candidate framework: C1-C6")
    print("AHP weights assigned: NO")
    print("Suitability score calculated: NO")
    print("=" * 65)

    if status != "PASS":
        raise SystemExit(
            "Data-readiness QA failed. Review checks before "
            "continuing."
        )


if __name__ == "__main__":
    main()
