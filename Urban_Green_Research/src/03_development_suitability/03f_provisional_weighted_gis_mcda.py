
"""
J26-DS-338 | Section 03 | Stage 3.8

PROVISIONAL AHP-WEIGHTED GIS-MCDA

Inputs:
- Stage 3.5 standardized criteria (CSV + GPKG)
- Stage 3.6/3.7 provisional AHP audit

Outputs:
- Continuous 0-1 suitability score (CSV + GPKG)
- One QA JSON report

IMPORTANT:
- Uses literature-adapted provisional AHP weights.
- NOT final expert-approved suitability.
- No urbanization pressure, SHAP, or conservation inputs.
- No suitability classes yet.
- No model training.
"""

from pathlib import Path
import json

import geopandas as gpd
import numpy as np
import pandas as pd
import pyogrio


ROOT = Path(__file__).resolve().parents[2]

STANDARD_DIR = (
    ROOT / "data/processed/development_suitability/"
    "standardized_criteria"
)

OUTPUT_DIR = (
    ROOT / "data/processed/development_suitability/"
    "weighted_suitability"
)

REPORT_DIR = (
    ROOT / "outputs/reports/development_suitability"
)

STANDARD_CSV = (
    STANDARD_DIR / "03d_standardized_criteria_100m.csv"
)

STANDARD_GPKG = (
    STANDARD_DIR / "03d_standardized_criteria_100m.gpkg"
)

AHP_AUDIT = REPORT_DIR / "03e_provisional_ahp_audit.json"

SCORE_COLUMNS = {
    "C1": "c1_road_access_score",
    "C2": "c2_road_context_score",
    "C3": "c3_development_context_score",
    "C4": "c4_slope_score",
}

WEIGHTED_SCORE = "provisional_suitability_score"


def load_and_validate_ahp():
    """Verify the previous AHP audit and recalculate weights."""

    audit = json.loads(
        AHP_AUDIT.read_text(encoding="utf-8")
    )

    if audit.get("status") != (
        "NUMERICALLY_CONSISTENT_PROVISIONAL_ONLY"
    ):
        raise ValueError("AHP audit has not passed")

    if audit.get("is_approved_final_ahp") is not False:
        raise ValueError(
            "Expected provisional AHP, not final approved AHP"
        )

    if audit.get("is_new_expert_assessment") is not False:
        raise ValueError(
            "AHP provenance does not match this workflow"
        )

    if audit.get("unsourced_scenario_comparisons") != 2:
        raise ValueError(
            "Unexpected AHP assumption provenance"
        )

    names = list(SCORE_COLUMNS)

    if audit.get("criteria_order") != names:
        raise ValueError("AHP criterion order mismatch")

    matrix = np.asarray(
        audit["pairwise_matrix"], dtype=float
    )

    if matrix.shape != (4, 4):
        raise ValueError("Expected a 4x4 AHP matrix")

    if not np.isfinite(matrix).all():
        raise ValueError("Nonfinite AHP matrix values")

    if (matrix <= 0).any():
        raise ValueError("AHP matrix must be positive")

    if not np.allclose(
        matrix * matrix.T, 1.0, atol=1e-10
    ):
        raise ValueError("AHP reciprocity check failed")

    eigvals, eigvecs = np.linalg.eig(matrix)

    idx = int(np.argmax(eigvals.real))

    principal = eigvals[idx]
    vector = eigvecs[:, idx]

    if abs(principal.imag) > 1e-9:
        raise ValueError("Complex principal eigenvalue")

    if np.max(np.abs(vector.imag)) > 1e-9:
        raise ValueError("Complex priority vector")

    weights = vector.real

    if weights.sum() < 0:
        weights = -weights

    if (weights <= 0).any():
        raise ValueError("Invalid AHP weights")

    weights = weights / weights.sum()

    saved_weights = np.array([
        audit["principal_eigenvector_weights"][key]
        for key in names
    ], dtype=float)

    if not np.allclose(
        weights, saved_weights, atol=1e-8
    ):
        raise ValueError(
            "Saved weights disagree with AHP matrix"
        )

    lambda_max = float(principal.real)
    ci = (lambda_max - 4) / 3
    cr = ci / 0.90

    if not np.isclose(
        cr, audit["consistency_ratio"], atol=1e-8
    ):
        raise ValueError(
            "Saved CR disagrees with recalculation"
        )

    if not (0 <= cr < 0.10):
        raise ValueError(
            f"AHP consistency unacceptable: CR={cr}"
        )

    return dict(zip(names, weights)), cr


def main():

    print("=" * 65)
    print("STAGE 3.8 - PROVISIONAL WEIGHTED GIS-MCDA")
    print("=" * 65)

    for path in [STANDARD_CSV, STANDARD_GPKG, AHP_AUDIT]:
        if not path.is_file():
            raise FileNotFoundError(path)

    # --------------------------------------------------
    # 1. AHP verification
    # --------------------------------------------------

    print("\n[1] VERIFY AHP WEIGHTS")

    weights, cr = load_and_validate_ahp()

    for name, weight in weights.items():
        print(f"{name}: {weight:.8f} ({weight*100:.4f}%)")

    print(f"Consistency Ratio: {cr:.8f}")
    print("Weight provenance: PROVISIONAL_LITERATURE_ADAPTED")

    # --------------------------------------------------
    # 2. Standardized input validation
    # --------------------------------------------------

    print("\n[2] LOAD STANDARDIZED CRITERIA")

    df = pd.read_csv(STANDARD_CSV)

    required = [
        "grid_id",
        "kmc_overlap_area_m2",
        "kmc_overlap_fraction",
        "boundary_cell",
        "coverage_group",
        "terrain_sample_method",
    ] + list(SCORE_COLUMNS.values())

    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(
            f"Missing standardized columns: {missing}"
        )

    if len(df) != 9240:
        raise ValueError("Expected exactly 9240 grid cells")

    if df.grid_id.isna().any() or not df.grid_id.is_unique:
        raise ValueError("Missing or duplicate grid IDs")

    input_columns = list(df.columns)

    forbidden = [
        c for c in input_columns
        if any(
            word in c.lower()
            for word in [
                "shap",
                "pressure",
                "conservation",
                "zoning",
            ]
        )
    ]

    if forbidden:
        raise ValueError(
            f"Forbidden input fields detected: {forbidden}"
        )

    selected = list(SCORE_COLUMNS.values())

    X = df[selected].to_numpy(dtype=float)

    if not np.isfinite(X).all():
        raise ValueError(
            "Missing or infinite standardized values"
        )

    if np.any(X < -1e-10) or np.any(X > 1 + 1e-10):
        raise ValueError(
            "Standardized criteria are outside [0, 1]"
        )

    print("Grid cells:", len(df))
    print("Standardized criteria:", len(selected))
    print("Missing values:", int(np.isnan(X).sum()))
    print("All values within 0-1: YES")

    # --------------------------------------------------
    # 3. Calculate weighted score
    # --------------------------------------------------

    print("\n[3] CALCULATE WEIGHTED SUITABILITY")

    w = np.array(
        [weights[name] for name in SCORE_COLUMNS],
        dtype=float,
    )

    contributions = X * w

    scores = contributions.sum(axis=1)

    result = df[required].copy()

    for index, name in enumerate(SCORE_COLUMNS):
        result[f"{name.lower()}_weighted_contribution"] = (
            contributions[:, index]
        )

    result[WEIGHTED_SCORE] = scores

    print("Score minimum:", round(float(scores.min()), 6))
    print("Score mean:", round(float(scores.mean()), 6))
    print("Score median:", round(float(np.median(scores)), 6))
    print(
        "Score 95th percentile:",
        round(float(np.quantile(scores, 0.95)), 6),
    )
    print("Score maximum:", round(float(scores.max()), 6))

    areas = result["kmc_overlap_area_m2"].to_numpy(
        dtype=float
    )

    if not np.isfinite(areas).all() or (areas <= 0).any():
        raise ValueError("Invalid KMC overlap areas")

    area_weighted_mean = float(
        np.average(scores, weights=areas)
    )

    print(
        "KMC area-weighted mean:",
        round(area_weighted_mean, 6),
    )

    print("\nHighest-scoring five cells:")

    print(
        result[
            ["grid_id", WEIGHTED_SCORE]
        ]
        .sort_values(
            WEIGHTED_SCORE, ascending=False
        )
        .head(5)
        .to_string(index=False)
    )

    # --------------------------------------------------
    # 4. GIS geometry integrity
    # --------------------------------------------------

    print("\n[4] VERIFY GIS GEOMETRY")

    geo_source = gpd.read_file(STANDARD_GPKG)

    if (
        geo_source.crs is None
        or geo_source.crs.to_epsg() != 32644
    ):
        raise ValueError("Invalid standardized GPKG CRS")

    if (
        len(geo_source) != 9240
        or geo_source.grid_id.isna().any()
        or not geo_source.grid_id.is_unique
    ):
        raise ValueError(
            "Unexpected standardized GIS grid IDs"
        )

    if (
        geo_source.geometry.isna().any()
        or not geo_source.geometry.is_valid.all()
    ):
        raise ValueError("Invalid GIS geometry")

    geo = geo_source[
        ["grid_id", "geometry"]
    ].merge(
        result,
        on="grid_id",
        how="inner",
        validate="one_to_one",
    )

    area_error = np.abs(
        geo.geometry.area.to_numpy(dtype=float)
        - geo["kmc_overlap_area_m2"].to_numpy(dtype=float)
    )

    max_area_error = float(area_error.max())

    print("GIS cells:", len(geo))
    print("GIS CRS:", geo.crs)
    print(
        "Maximum grid/KMC area error:",
        round(max_area_error, 8),
        "m2",
    )

    # --------------------------------------------------
    # 5. Validation
    # --------------------------------------------------

    print("\n[5] FINAL QA")

    recomputed = X @ w

    checks = {
        "9240_cells_preserved": len(result) == 9240,
        "unique_ids": result.grid_id.is_unique,
        "same_csv_gis_ids": (
            len(geo) == 9240
            and set(result.grid_id) == set(geo.grid_id)
        ),
        "four_standardized_inputs": X.shape == (9240, 4),
        "weights_sum_one": bool(
            np.isclose(w.sum(), 1.0)
        ),
        "weights_positive": bool((w > 0).all()),
        "all_scores_finite": bool(
            np.isfinite(scores).all()
        ),
        "all_scores_in_0_1": bool(
            ((scores >= -1e-10) & (scores <= 1 + 1e-10)).all()
        ),
        "weighted_sum_reproduced": bool(
            np.allclose(
                scores, recomputed, atol=1e-12
            )
        ),
        "gis_area_reconciled": (
            max_area_error <= 1.0
        ),
        "expected_cr": 0 <= cr < 0.10,
    }

    for name, passed in checks.items():
        print(
            f"[{'PASS' if passed else 'FAIL'}] {name}"
        )

    if not all(checks.values()):
        raise SystemExit(
            "Stage 3.8 QA failed. No outputs exported."
        )

    # --------------------------------------------------
    # 6. Export compact evidence package
    # --------------------------------------------------

    print("\n[6] EXPORT RESULTS")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    csv_path = (
        OUTPUT_DIR /
        "03f_provisional_weighted_suitability_100m.csv"
    )

    gpkg_path = (
        OUTPUT_DIR /
        "03f_provisional_weighted_suitability_100m.gpkg"
    )

    qa_path = (
        REPORT_DIR /
        "03f_provisional_weighted_suitability_qa.json"
    )

    result.to_csv(csv_path, index=False)

    geo.to_file(
        gpkg_path,
        layer="provisional_weighted_suitability_100m",
        driver="GPKG",
    )

    csv_check = pd.read_csv(csv_path)

    gpkg_info = pyogrio.read_info(gpkg_path)

    export_pass = (
        len(csv_check) == 9240
        and csv_check.grid_id.is_unique
        and gpkg_info["features"] == 9240
        and np.allclose(
            csv_check[WEIGHTED_SCORE].to_numpy(),
            scores,
            atol=1e-10,
        )
    )

    checks["export_verification"] = bool(export_pass)

    status = (
        "PASS_PROVISIONAL"
        if all(checks.values())
        else "REVIEW_REQUIRED"
    )

    report = {
        "stage": "3.8",
        "status": status,
        "method": "AHP weighted linear combination",
        "weight_provenance":
            "PROVISIONAL_LITERATURE_ADAPTED",
        "final_expert_ahp_completed": False,
        "is_final_approved_suitability": False,
        "criteria_weights": weights,
        "ahp_consistency_ratio": cr,
        "grid_cells": len(result),
        "study_area_km2": float(areas.sum() / 1e6),
        "score_statistics": {
            "min": float(scores.min()),
            "mean": float(scores.mean()),
            "median": float(np.median(scores)),
            "p05": float(np.quantile(scores, 0.05)),
            "p95": float(np.quantile(scores, 0.95)),
            "max": float(scores.max()),
            "area_weighted_mean": area_weighted_mean,
        },
        "max_geometry_area_error_m2": max_area_error,
        "checks": checks,
        "limitations": [
            "Provisional literature-informed AHP scenario.",
            "Two pairwise judgments are scenario assumptions.",
            "Not verified by a Kaduwela planning expert.",
            "Relative physical/accessibility suitability only.",
            "No legal development permission implied.",
            "No suitability classes calculated yet.",
            "Sensitivity analysis is still required.",
            "Final approved weights remain pending.",
        ],
    }

    qa_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print("CSV:", csv_path)
    print("GPKG:", gpkg_path)
    print("QA:", qa_path)

    print("\n" + "=" * 65)
    print("STAGE 3.8:", status)
    print("Grid cells:", len(result))
    print("Suitability score scale: 0-1")
    print("Suitability classes: NOT YET")
    print("Final expert approval: PENDING")
    print("=" * 65)

    if status != "PASS_PROVISIONAL":
        raise SystemExit(
            "Stage 3.8 export verification failed."
        )


if __name__ == "__main__":
    main()
