
"""
J26-DS-338 | Section 03
Stage 16C-M - Final Selected-Criteria Standardization

SOURCE:
    16b_raw_suitability_criteria_100m

SELECTED CRITERIA:
    C1 - Major Road Proximity (COST)
    C2 - Local Road Density, 500m (BENEFIT)
    C4 - Terrain Slope (COST)

METHOD:
    Empirical 5th-95th percentile normalization,
    clipped to [0,1].

SAFETY:
    - Verify source 16B QA.
    - Reproduce old 03D standardization method.
    - Preserve source files.
    - Do not overwrite earlier outputs.
    - Do not assign AHP weights at this stage.
    - Do not modify Section 04.
"""

from pathlib import Path
from tempfile import TemporaryDirectory
import hashlib
import json

import geopandas as gpd
import numpy as np
import pandas as pd


# ============================================================
# 1. PROJECT PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

BASE = (
    ROOT
    / "data"
    / "processed"
    / "development_suitability"
)

REPORT = ROOT / "outputs" / "reports"

RAW_CSV = (
    BASE
    / "raw_criteria"
    / "16b_raw_suitability_criteria_100m.csv"
)

RAW_GPKG = (
    BASE
    / "raw_criteria"
    / "16b_raw_suitability_criteria_100m.gpkg"
)

RAW_LAYER = "raw_suitability_criteria_100m"

RAW_QA = (
    REPORT
    / "16b_raw_suitability_criteria_summary.json"
)

OLD_STD = (
    BASE
    / "standardized_criteria"
    / "03d_standardized_criteria_100m.csv"
)

OUT_DIR = BASE / "standardized_criteria"

OUT_CSV = (
    OUT_DIR
    / "16c_m_final_criteria_standardized_100m.csv"
)

OUT_GPKG = (
    OUT_DIR
    / "16c_m_final_criteria_standardized_100m.gpkg"
)

OUT_LAYER = "final_criteria_standardized_100m"

OUT_QA = (
    REPORT
    / "16c_m_final_criteria_standardization_qa.json"
)


# ============================================================
# 2. FROZEN CRITERIA CONFIGURATION
# ============================================================

CRITERIA = {
    "C1": {
        "name": "Major Road Proximity",
        "raw": "c1_major_road_distance_m",
        "score": "c1_road_access_score",
        "direction": "COST",
    },
    "C2": {
        "name": "Local Road Density 500m",
        "raw": "c2_road_density_500m_km_per_km2",
        "score": "c2_road_context_score",
        "direction": "BENEFIT",
    },
    "C4": {
        "name": "Terrain Slope",
        "raw": "c4_slope_mean_deg",
        "score": "c4_slope_score",
        "direction": "COST",
    },
}

METADATA_COLUMNS = [
    "grid_id",
    "kmc_overlap_area_m2",
    "kmc_overlap_fraction",
    "boundary_cell",
    "coverage_group",
    "terrain_fallback_applied",
    "terrain_extraction_method",
    "terrain_fallback_distance_m",
]


# ============================================================
# 3. HELPERS
# ============================================================

def file_hash(path):

    h = hashlib.sha256()

    with path.open("rb") as f:

        for chunk in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(chunk)

    return h.hexdigest()


def require(condition, description):

    passed = bool(condition)

    print(
        f"[{'PASS' if passed else 'FAIL'}] "
        f"{description}"
    )

    if not passed:

        raise ValueError(
            f"Quality assurance failed: {description}"
        )


def validate_ids(frame, source_name):

    require(
        len(frame) == 9240,
        f"{source_name}: 9,240 grid cells",
    )

    require(
        "grid_id" in frame.columns
        and frame["grid_id"].notna().all()
        and frame["grid_id"].is_unique,
        f"{source_name}: unique nonmissing grid IDs",
    )


def values_of(frame, column):

    if column not in frame.columns:

        raise KeyError(
            f"Missing required column: {column}"
        )

    return pd.to_numeric(
        frame[column],
        errors="coerce",
    ).to_numpy(dtype=float)


def standardize(values, lower, upper, direction):

    if (
        not np.isfinite(lower)
        or not np.isfinite(upper)
        or upper <= lower
    ):

        raise ValueError(
            "Invalid percentile normalization anchors"
        )

    if direction == "BENEFIT":

        scores = (
            (values - lower)
            / (upper - lower)
        )

    elif direction == "COST":

        scores = (
            (upper - values)
            / (upper - lower)
        )

    else:

        raise ValueError(
            f"Unknown direction: {direction}"
        )

    return np.clip(
        scores,
        0.0,
        1.0,
    )


def calculate_scores(values, direction):

    lower = float(
        np.quantile(values, 0.05)
    )

    upper = float(
        np.quantile(values, 0.95)
    )

    scores = standardize(
        values,
        lower,
        upper,
        direction,
    )

    return scores, lower, upper


def check_monotonicity(values, scores, direction):

    order = np.argsort(
        values,
        kind="stable",
    )

    differences = np.diff(
        scores[order]
    )

    tolerance = 1e-10

    if direction == "BENEFIT":

        return bool(
            np.all(
                differences >= -tolerance
            )
        )

    return bool(
        np.all(
            differences <= tolerance
        )
    )


def describe(values):

    v = np.asarray(
        values,
        dtype=float,
    )

    return {
        "count": int(len(v)),
        "min": float(np.min(v)),
        "p05": float(np.quantile(v, 0.05)),
        "median": float(np.median(v)),
        "mean": float(np.mean(v)),
        "p95": float(np.quantile(v, 0.95)),
        "max": float(np.max(v)),
    }


# ============================================================
# 4. MAIN WORKFLOW
# ============================================================

def main():

    print("=" * 78)
    print(
        "SECTION 03 | STAGE 16C-M "
        "FINAL CRITERIA STANDARDIZATION"
    )
    print("=" * 78)

    inputs = [
        RAW_CSV,
        RAW_GPKG,
        RAW_QA,
        OLD_STD,
    ]

    outputs = [
        OUT_CSV,
        OUT_GPKG,
        OUT_QA,
    ]

    for path in inputs:

        if not path.is_file():

            raise FileNotFoundError(path)

    for path in outputs:

        if path.exists():

            raise FileExistsError(
                "Existing output must not be overwritten: "
                f"{path}"
            )

    input_hashes = {
        path.relative_to(ROOT).as_posix(): file_hash(path)
        for path in inputs
    }

    # ========================================================
    # STEP 1 - VALIDATE STAGE 16B
    # ========================================================

    print("\n[1] VERIFY LATEST RAW SOURCE")

    source_qa = json.loads(
        RAW_QA.read_text(
            encoding="utf-8"
        )
    )

    require(
        source_qa.get(
            "overall_step16b_status"
        ) == "PASS",
        "16B raw source QA = PASS",
    )

    require(
        source_qa.get(
            "analysis_grid_cells"
        ) == 9240,
        "16B analysis grid count",
    )

    require(
        source_qa.get(
            "analysis_crs"
        ) == "EPSG:32644",
        "16B coordinate reference system",
    )

    require(
        source_qa.get(
            "original_valid_terrain_preserved"
        ) is True,
        "Original valid terrain measurements preserved",
    )

    actual_fallback = float(
        source_qa[
            "terrain_fallback_max_actual_distance_m"
        ]
    )

    allowed_fallback = float(
        source_qa[
            "terrain_fallback_max_allowed_distance_m"
        ]
    )

    require(
        actual_fallback <= allowed_fallback,
        "Terrain fallback within documented distance limit",
    )

    require(
        source_qa.get(
            "ahp_weights_assigned"
        ) is False,
        "16B remains raw-criteria stage",
    )

    require(
        source_qa.get(
            "standardization_performed"
        ) is False,
        "16B has not already standardized its criteria",
    )

    require(
        source_qa.get(
            "suitability_score_calculated"
        ) is False,
        "16B has not calculated suitability scores",
    )

    print(
        "Terrain fallback cells:",
        source_qa.get("terrain_fallback_count"),
    )

    print(
        "Maximum fallback distance:",
        actual_fallback,
        "m",
    )

    # ========================================================
    # STEP 2 - LOAD GRID DATA
    # ========================================================

    print("\n[2] LOAD AND VALIDATE RAW DATA")

    raw = pd.read_csv(
        RAW_CSV
    )

    old = pd.read_csv(
        OLD_STD
    )

    validate_ids(
        raw,
        "16B raw CSV",
    )

    validate_ids(
        old,
        "03D historical standardized CSV",
    )

    require(
        set(raw["grid_id"]) == set(old["grid_id"]),
        "16B and 03D grid identities match",
    )

    required_columns = (
        METADATA_COLUMNS
        + [c["raw"] for c in CRITERIA.values()]
    )

    missing_columns = [
        column
        for column in required_columns
        if column not in raw.columns
    ]

    require(
        not missing_columns,
        "All required raw and metadata fields exist",
    )

    require(
        np.isfinite(
            values_of(
                raw,
                "kmc_overlap_area_m2",
            )
        ).all(),
        "KMC area values are finite",
    )

    require(
        np.isclose(
            values_of(
                raw,
                "kmc_overlap_area_m2",
            ).sum() / 1e6,
            88.82921581108494,
            atol=0.001,
            rtol=0,
        ),
        "KMC grid areas sum to verified study area",
    )

    # ========================================================
    # STEP 3 - REPRODUCE ORIGINAL 03D METHOD
    # ========================================================

    print(
        "\n[3] VERIFY ORIGINAL NORMALIZATION METHOD"
    )

    old_indexed = old.set_index(
        "grid_id"
    )

    old_indexed = old_indexed.loc[
        raw["grid_id"]
    ]

    historical_reproduction = {}

    for code, config in CRITERIA.items():

        raw_column = config["raw"]
        score_column = config["score"]
        direction = config["direction"]

        historical_raw = values_of(
            old_indexed,
            raw_column,
        )

        historical_score = values_of(
            old_indexed,
            score_column,
        )

        require(
            np.isfinite(historical_raw).all()
            and np.isfinite(historical_score).all(),
            f"{code}: historical measurements finite",
        )

        reproduced, lower, upper = (
            calculate_scores(
                historical_raw,
                direction,
            )
        )

        maximum_difference = float(
            np.max(
                np.abs(
                    reproduced - historical_score
                )
            )
        )

        require(
            np.allclose(
                reproduced,
                historical_score,
                atol=1e-9,
                rtol=0,
            ),
            f"{code}: original 03D scores reproduced",
        )

        historical_reproduction[code] = {
            "lower_p05": lower,
            "upper_p95": upper,
            "maximum_score_difference": maximum_difference,
        }

        print(
            f"{code}: old standardization reproduced; "
            f"max error = {maximum_difference:.12f}"
        )

    # ========================================================
    # STEP 4 - NEW STANDARDIZATION
    # ========================================================

    print(
        "\n[4] STANDARDIZE SELECTED 16B CRITERIA"
    )

    result = raw[
        required_columns
    ].copy()

    parameters = {}

    for code, config in CRITERIA.items():

        raw_column = config["raw"]
        score_column = config["score"]
        direction = config["direction"]

        measurements = values_of(
            raw,
            raw_column,
        )

        require(
            np.isfinite(measurements).all(),
            f"{code}: 16B raw measurements finite",
        )

        scores, lower, upper = (
            calculate_scores(
                measurements,
                direction,
            )
        )

        require(
            np.isfinite(scores).all(),
            f"{code}: standardized scores finite",
        )

        require(
            bool(
                np.all(scores >= 0)
                and np.all(scores <= 1)
            ),
            f"{code}: score range [0, 1]",
        )

        require(
            check_monotonicity(
                measurements,
                scores,
                direction,
            ),
            f"{code}: correct benefit/cost direction",
        )

        result[score_column] = scores

        parameters[code] = {
            "name": config["name"],
            "raw_column": raw_column,
            "score_column": score_column,
            "direction": direction,
            "lower_p05": lower,
            "upper_p95": upper,
            "raw_distribution": describe(
                measurements
            ),
            "score_distribution": describe(
                scores
            ),
        }

        print(
            f"\n{code} - {config['name']}"
        )

        print(
            f"  Direction: {direction}"
        )

        print(
            f"  P05: {lower:.8f}"
        )

        print(
            f"  P95: {upper:.8f}"
        )

        print(
            f"  Score min/max: "
            f"{scores.min():.6f} / "
            f"{scores.max():.6f}"
        )

    result["standardization_method"] = (
        "P05_P95_CLIPPED_MINMAX"
    )

    result["standardization_status"] = (
        "PASS_SELECTED_CRITERIA_STANDARDIZED"
    )

    result["selected_criteria_set"] = (
        "C1_C2_C4"
    )

    # Not an AHP-weighted result.
    # Weights will be applied in the next stage.

    require(
        len(result) == 9240
        and result["grid_id"].is_unique,
        "Standardized output has 9,240 unique cells",
    )

    require(
        "development_suitability_score"
        not in result.columns,
        "No premature final suitability score",
    )

    # ========================================================
    # STEP 5 - VERIFY NEW RAW GIS GEOMETRY
    # ========================================================

    print("\n[5] VERIFY LATEST GIS SOURCE")

    geometry = gpd.read_file(
        RAW_GPKG,
        layer=RAW_LAYER,
    )

    validate_ids(
        geometry,
        "16B raw GeoPackage",
    )

    require(
        geometry.crs is not None
        and geometry.crs.to_epsg() == 32644,
        "GIS coordinate system EPSG:32644",
    )

    require(
        geometry.geometry.notna().all()
        and not geometry.geometry.is_empty.any(),
        "All grid geometries exist",
    )

    require(
        geometry.geometry.is_valid.all(),
        "All grid geometries valid",
    )

    require(
        set(geometry["grid_id"])
        == set(raw["grid_id"]),
        "GIS and CSV grid IDs match",
    )

    geometry_indexed = geometry.set_index(
        "grid_id"
    ).loc[
        raw["grid_id"]
    ]

    for code, config in CRITERIA.items():

        column = config["raw"]

        require(
            column in geometry_indexed.columns,
            f"{code}: source GIS raw field exists",
        )

        csv_values = values_of(
            raw,
            column,
        )

        gis_values = values_of(
            geometry_indexed,
            column,
        )

        require(
            np.allclose(
                csv_values,
                gis_values,
                atol=1e-6,
                rtol=1e-9,
                equal_nan=False,
            ),
            f"{code}: GIS/CSV measurements agree",
        )

    area_values = values_of(
        geometry_indexed,
        "kmc_overlap_area_m2",
    )

    actual_areas = (
        geometry_indexed.geometry.area
        .to_numpy(dtype=float)
    )

    require(
        np.allclose(
            area_values,
            actual_areas,
            atol=1.0,
            rtol=0,
        ),
        "GIS geometry and KMC cell areas agree",
    )

    require(
        np.isclose(
            actual_areas.sum() / 1e6,
            88.82921581108494,
            atol=0.001,
            rtol=0,
        ),
        "GIS covers the expected KMC analysis area",
    )

    result_geo = geometry[
        ["grid_id", "geometry"]
    ].merge(
        result,
        on="grid_id",
        how="inner",
        validate="one_to_one",
    )

    require(
        len(result_geo) == 9240
        and result_geo["grid_id"].is_unique,
        "Final GIS score join is one-to-one",
    )

    # ========================================================
    # STEP 6 - EXPORT WITHOUT OVERWRITING
    # ========================================================

    print("\n[6] WRITE AND VERIFY NEW OUTPUTS")

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with TemporaryDirectory(
        prefix="16c_m_build_",
        dir=OUT_DIR,
    ) as temporary:

        temporary = Path(temporary)

        temp_csv = temporary / OUT_CSV.name
        temp_gpkg = temporary / OUT_GPKG.name

        result.to_csv(
            temp_csv,
            index=False,
        )

        result_geo.to_file(
            temp_gpkg,
            layer=OUT_LAYER,
            driver="GPKG",
            engine="pyogrio",
        )

        check_csv = pd.read_csv(
            temp_csv
        )

        check_geo = gpd.read_file(
            temp_gpkg,
            layer=OUT_LAYER,
        )

        validate_ids(
            check_csv,
            "Exported CSV",
        )

        validate_ids(
            check_geo,
            "Exported GeoPackage",
        )

        require(
            check_geo.crs is not None
            and check_geo.crs.to_epsg() == 32644,
            "Exported GIS CRS correct",
        )

        original_scores = result.set_index(
            "grid_id"
        )

        checked_scores = check_csv.set_index(
            "grid_id"
        ).loc[
            original_scores.index
        ]

        for code, config in CRITERIA.items():

            column = config["score"]

            require(
                np.allclose(
                    values_of(
                        original_scores,
                        column,
                    ),
                    values_of(
                        checked_scores,
                        column,
                    ),
                    atol=1e-12,
                    rtol=0,
                ),
                f"{code}: exported CSV score preserved",
            )

        # Recheck outputs immediately before moving.
        for path in outputs:

            if path.exists():

                raise FileExistsError(
                    f"Output appeared during processing: {path}"
                )

        temp_csv.replace(
            OUT_CSV
        )

        temp_gpkg.replace(
            OUT_GPKG
        )

    # ========================================================
    # STEP 7 - INPUT INTEGRITY
    # ========================================================

    print("\n[7] VERIFY SOURCE INTEGRITY")

    for path in inputs:

        relative = path.relative_to(
            ROOT
        ).as_posix()

        require(
            file_hash(path)
            == input_hashes[relative],
            f"Input unchanged: {path.name}",
        )

    # ========================================================
    # STEP 8 - REPORT
    # ========================================================

    report = {
        "section": "03_development_suitability",
        "stage": "16c_m",
        "status": (
            "PASS_FINAL_SELECTED_CRITERIA_STANDARDIZED_"
            "AHP_WEIGHTING_PENDING"
        ),
        "raw_source": RAW_CSV.relative_to(
            ROOT
        ).as_posix(),
        "raw_source_step16b_status": (
            source_qa["overall_step16b_status"]
        ),
        "grid_cells": 9240,
        "crs": "EPSG:32644",
        "study_area_km2": float(
            actual_areas.sum() / 1e6
        ),
        "selected_criteria": [
            "C1", "C2", "C4"
        ],
        "method": (
            "Empirical P05/P95 clipped normalization"
        ),
        "historical_03d_method_reproduced": True,
        "historical_reproduction": (
            historical_reproduction
        ),
        "criteria": parameters,
        "terrain_fallback_count": (
            source_qa["terrain_fallback_count"]
        ),
        "terrain_fallback_method": (
            source_qa["terrain_fallback_method"]
        ),
        "ahp_weights_applied": False,
        "final_suitability_calculated": False,
        "classification_performed": False,
        "frozen_section04_modified": False,
        "inputs_sha256": input_hashes,
        "outputs": {
            "csv": OUT_CSV.relative_to(
                ROOT
            ).as_posix(),
            "gpkg": OUT_GPKG.relative_to(
                ROOT
            ).as_posix(),
            "csv_sha256": file_hash(
                OUT_CSV
            ),
            "gpkg_sha256": file_hash(
                OUT_GPKG
            ),
        },
    }

    OUT_QA.write_text(
        json.dumps(
            report,
            indent=2,
        ),
        encoding="utf-8",
    )

    # ========================================================
    # STEP 9 - TERMINAL SUMMARY
    # ========================================================

    print("\n" + "=" * 78)

    print(
        "PASS_FINAL_SELECTED_CRITERIA_STANDARDIZED_"
        "AHP_WEIGHTING_PENDING"
    )

    print("Grid cells:", len(result))

    print(
        "Criteria:",
        list(CRITERIA)
    )

    print(
        "Normalization:",
        "P05/P95 clipped 0-1",
    )

    print(
        "Historical normalization reproduced: YES"
    )

    print(
        "Original standardized files modified: NO"
    )

    print(
        "Expert AHP weights applied: NO - NEXT STAGE"
    )

    print(
        "Section 04 files modified: NO"
    )

    print("\nCSV:", OUT_CSV)
    print("GPKG:", OUT_GPKG)
    print("QA:", OUT_QA)

    print("=" * 78)


if __name__ == "__main__":
    main()
