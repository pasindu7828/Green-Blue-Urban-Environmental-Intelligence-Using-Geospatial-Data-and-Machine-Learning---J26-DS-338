
"""
J26-DS-338 | Section 03 | Stage 3.11
Consolidated Provisional Technical Release Audit

Rechecks existing GIS-MCDA outputs across stages.
Generates one JSON audit and one viva summary.
Does NOT certify approved expert AHP.
"""

from pathlib import Path
import hashlib
import json

import geopandas as gpd
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data/processed/development_suitability"
REPORT = ROOT / "outputs/reports/development_suitability"

MAP = (
    ROOT / "outputs/maps/development_suitability/"
    "03g_provisional_suitability_map.png"
)

BOUNDARY = (
    ROOT / "data/processed/boundaries/kmc/"
    "kaduwela_kmc_boundary_utm44n.gpkg"
)

FILES = {
    "raw": DATA / "raw_criteria/03b_raw_criteria_100m.csv",
    "standard": (
        DATA / "standardized_criteria/"
        "03d_standardized_criteria_100m.csv"
    ),
    "weighted": (
        DATA / "weighted_suitability/"
        "03f_provisional_weighted_suitability_100m.csv"
    ),
    "classified": (
        DATA / "weighted_suitability/"
        "03g_provisional_suitability_classes_100m.csv"
    ),
    "classified_gis": (
        DATA / "weighted_suitability/"
        "03g_provisional_suitability_classes_100m.gpkg"
    ),
    "sensitivity": REPORT / "03h_sensitivity_scenarios.csv",
    "map": MAP,
}

AUDITS = {
    "3.1-3.2": (
        "03_data_readiness_qa.json",
        "status", "PASS"
    ),
    "3.3": (
        "03b_raw_criteria_qa.json",
        "status", "PASS"
    ),
    "3.4": (
        "03c_analysis_qa.json",
        "status", "PASS"
    ),
    "3.5": (
        "03d_standardization_qa.json",
        "qa_status", "PASS"
    ),
    "3.6-3.7": (
        "03e_provisional_ahp_audit.json",
        "status",
        "NUMERICALLY_CONSISTENT_PROVISIONAL_ONLY"
    ),
    "3.8": (
        "03f_provisional_weighted_suitability_qa.json",
        "status", "PASS_PROVISIONAL"
    ),
    "3.9": (
        "03g_suitability_classification_qa.json",
        "status", "PASS_PROVISIONAL"
    ),
    "3.10": (
        "03h_sensitivity_qa.json",
        "status", "PASS_PROVISIONAL"
    ),
}

SCORE = "provisional_suitability_score"

COLUMNS = [
    "c1_road_access_score",
    "c2_road_context_score",
    "c3_development_context_score",
    "c4_slope_score",
]

CLASS_NAMES = [
    "VERY_LOW",
    "LOW",
    "MODERATE",
    "HIGH",
    "VERY_HIGH",
]


def sha256(path):
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(
            lambda: file.read(1024 * 1024), b""
        ):
            digest.update(chunk)

    return digest.hexdigest()


def load_csv(path):
    df = pd.read_csv(path)

    if (
        "grid_id" not in df
        or len(df) != 9240
        or df.grid_id.isna().any()
        or not df.grid_id.is_unique
    ):
        raise ValueError(
            f"Grid ID/row count invalid: {path.name}"
        )

    return df.set_index("grid_id").sort_index()


def main():

    print("=" * 64)
    print("STAGE 3.11 - PROVISIONAL TECHNICAL RELEASE AUDIT")
    print("=" * 64)

    required_files = [
        *FILES.values(),
        BOUNDARY,
        *(
            REPORT / name
            for name, _, _ in AUDITS.values()
        ),
    ]

    for path in required_files:
        if not path.is_file():
            raise FileNotFoundError(path)

    checks = {}
    evidence = {}

    # -----------------------------------------------
    # 1. Earlier stage QA verification
    # -----------------------------------------------

    print("\n[1] VERIFY EARLIER STAGE AUDITS")

    for stage, (filename, field, expected) in AUDITS.items():

        audit = json.loads(
            (REPORT / filename).read_text(
                encoding="utf-8"
            )
        )

        key = (
            "upstream_"
            + stage.replace(".", "_").replace("-", "_")
        )

        checks[key] = audit.get(field) == expected
        evidence[stage] = audit.get(field)

        print(
            f"[{'PASS' if checks[key] else 'FAIL'}] "
            f"Stage {stage}: {audit.get(field)}"
        )

    # -----------------------------------------------
    # 2. AHP mathematical reproduction
    # -----------------------------------------------

    print("\n[2] REPRODUCE AHP AND SUITABILITY")

    ahp = json.loads(
        (
            REPORT / AUDITS["3.6-3.7"][0]
        ).read_text(encoding="utf-8")
    )

    matrix = np.asarray(
        ahp["pairwise_matrix"], dtype=float
    )

    names = ["C1", "C2", "C3", "C4"]

    checks["ahp_provisional_not_expert_approved"] = (
        ahp.get("is_approved_final_ahp") is False
        and ahp.get("is_new_expert_assessment") is False
        and ahp.get("unsourced_scenario_comparisons") == 2
    )

    checks["ahp_reciprocal"] = (
        matrix.shape == (4, 4)
        and np.isfinite(matrix).all()
        and (matrix > 0).all()
        and np.allclose(matrix * matrix.T, 1.0)
    )

    if not checks["ahp_reciprocal"]:
        raise SystemExit("Invalid AHP matrix")

    eigenvalues, eigenvectors = np.linalg.eig(matrix)
    index = int(np.argmax(eigenvalues.real))

    principal = eigenvalues[index]
    vector = eigenvectors[:, index]

    if (
        abs(principal.imag) > 1e-9
        or np.max(np.abs(vector.imag)) > 1e-9
    ):
        raise ValueError("Unexpected complex AHP eigenpair")

    weights = vector.real

    if weights.sum() < 0:
        weights = -weights

    weights /= weights.sum()

    cr = float(
        (principal.real - 4) / 3 / 0.90
    )

    saved_weights = np.array([
        ahp["principal_eigenvector_weights"][name]
        for name in names
    ])

    checks["ahp_weights_reproduced"] = bool(
        (weights > 0).all()
        and np.allclose(
            weights, saved_weights, atol=1e-10
        )
        and np.isclose(weights.sum(), 1.0)
    )

    checks["ahp_cr_reproduced_below_0_10"] = bool(
        0 <= cr < 0.10
        and np.isclose(
            cr, ahp["consistency_ratio"], atol=1e-10
        )
    )

    # -----------------------------------------------
    # 3. Data consistency
    # -----------------------------------------------

    raw = load_csv(FILES["raw"])
    standard = load_csv(FILES["standard"])
    weighted = load_csv(FILES["weighted"])
    classified = load_csv(FILES["classified"])

    checks["all_four_csv_grid_id_sets_match"] = (
        raw.index.equals(standard.index)
        and standard.index.equals(weighted.index)
        and weighted.index.equals(classified.index)
    )

    areas = classified[
        "kmc_overlap_area_m2"
    ].to_numpy(dtype=float)

    X = standard[COLUMNS].to_numpy(dtype=float)

    observed = weighted[
        SCORE
    ].to_numpy(dtype=float)

    observed_class = classified[
        "suitability_class_code"
    ].to_numpy(dtype=int)

    predicted_class = np.digitize(
        observed, [0.2, 0.4, 0.6, 0.8]
    ) + 1

    checks["area_positive_finite"] = bool(
        np.isfinite(areas).all()
        and (areas > 0).all()
    )

    checks["standardized_criteria_in_0_1"] = bool(
        np.isfinite(X).all()
        and (X >= 0).all()
        and (X <= 1).all()
    )

    checks["continuous_scores_reproduced"] = bool(
        np.isfinite(observed).all()
        and (observed >= 0).all()
        and (observed <= 1).all()
        and np.allclose(
            X @ weights, observed, atol=1e-10
        )
        and np.allclose(
            classified[SCORE].to_numpy(dtype=float),
            observed,
            atol=1e-10,
        )
    )

    checks["five_class_labels_reproduced"] = bool(
        np.array_equal(
            observed_class, predicted_class
        )
        and np.array_equal(
            classified[
                "suitability_class"
            ].to_numpy(dtype=str),
            np.array(CLASS_NAMES)[predicted_class - 1],
        )
    )

    # -----------------------------------------------
    # 4. GIS geometry and area checks
    # -----------------------------------------------

    print("\n[3] GIS AND STUDY-AREA QA")

    geo = gpd.read_file(FILES["classified_gis"])
    kmc = gpd.read_file(BOUNDARY)

    checks["gis_9240_valid_unique_cells"] = bool(
        len(geo) == 9240
        and geo.grid_id.is_unique
        and geo.grid_id.notna().all()
        and geo.geometry.notna().all()
        and geo.geometry.is_valid.all()
    )

    checks["gis_and_boundary_epsg32644"] = bool(
        geo.crs is not None
        and geo.crs.to_epsg() == 32644
        and kmc.crs is not None
        and kmc.crs.to_epsg() == 32644
    )

    joined = (
        geo.set_index("grid_id")
        .loc[classified.index]
    )

    area_error = float(
        np.max(
            np.abs(
                joined.geometry.area.to_numpy()
                - areas
            )
        )
    )

    kmc_total = float(kmc.geometry.area.sum())

    checks["gis_areas_reconcile"] = bool(
        area_error <= 1.0
        and abs(float(areas.sum()) - kmc_total) <= 1.0
    )

    checks["gis_scores_and_classes_match_csv"] = bool(
        np.allclose(
            joined[SCORE].to_numpy(dtype=float),
            observed,
            atol=1e-10,
        )
        and np.array_equal(
            joined[
                "suitability_class_code"
            ].to_numpy(dtype=int),
            observed_class,
        )
    )

    checks["map_png_nonempty"] = bool(
        MAP.stat().st_size > 10000
    )

    # -----------------------------------------------
    # 5. Sensitivity evidence checks
    # -----------------------------------------------

    print("\n[4] SENSITIVITY EVIDENCE QA")

    sensitivity = pd.read_csv(FILES["sensitivity"])

    checks["nineteen_sensitivity_scenarios"] = bool(
        len(sensitivity) == 19
        and sensitivity.scenario.is_unique
    )

    metrics = [
        "spearman_rank_vs_baseline",
        "class_changed_kmc_area_pct",
        "high_very_high_area_pct",
        "high_area_overlap_jaccard",
    ]

    checks["finite_sensitivity_metrics"] = bool(
        np.isfinite(
            sensitivity[metrics].to_numpy(dtype=float)
        ).all()
    )

    inconsistent = sensitivity.loc[
        sensitivity["ahp_cr"].ge(0.10),
        "scenario",
    ].tolist()

    high = observed_class >= 4

    high_pct = (
        100.0 * float(areas[high].sum() / areas.sum())
    )

    sensitivity_qa = json.loads(
        (
            REPORT / AUDITS["3.10"][0]
        ).read_text(encoding="utf-8")
    )

    checks["high_area_agrees_with_stage_3_10"] = bool(
        np.isclose(
            high_pct,
            sensitivity_qa[
                "baseline_high_very_high_area_pct"
            ],
            atol=1e-8,
        )
    )

    for name, passed in checks.items():
        print(
            f"[{'PASS' if passed else 'FAIL'}] {name}"
        )

    if not all(checks.values()):
        print("STAGE 3.11: REVIEW_REQUIRED")
        raise SystemExit(
            "Final technical audit failed. "
            "No release reports generated."
        )

    # -----------------------------------------------
    # 6. Build compact final audit
    # -----------------------------------------------

    print("\n[5] GENERATE PROVISIONAL RELEASE REPORTS")

    inventory = {}

    for name, path in FILES.items():
        inventory[name] = {
            "path": str(path.relative_to(ROOT)),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }

    class_area = []

    for i, name in enumerate(CLASS_NAMES, start=1):

        mask = observed_class == i

        class_area.append({
            "class": name,
            "count": int(mask.sum()),
            "area_km2": float(
                areas[mask].sum() / 1e6
            ),
            "kmc_area_pct": float(
                100 * areas[mask].sum() / areas.sum()
            ),
        })

    results = {
        "stage": "3.11",
        "technical_status": "PASS_PROVISIONAL",
        "methodology_status": (
            "NOT_FINAL_EXPERT_AHP_PENDING"
        ),
        "approved_expert_judgments": False,
        "study_area_km2": kmc_total / 1e6,
        "grid_cells": 9240,
        "ahp_weights": dict(
            zip(
                names,
                [float(value) for value in weights],
            )
        ),
        "ahp_consistency_ratio": cr,
        "ahp_unverified_scenario_pairs": 2,
        "score_mean": float(observed.mean()),
        "score_area_weighted_mean": float(
            np.average(observed, weights=areas)
        ),
        "high_very_high_area_pct": high_pct,
        "class_area": class_area,
        "sensitivity_scenarios": len(sensitivity),
        "minimum_sensitivity_spearman": float(
            sensitivity[
                "spearman_rank_vs_baseline"
            ].min()
        ),
        "maximum_changed_kmc_area_pct": float(
            sensitivity[
                "class_changed_kmc_area_pct"
            ].max()
        ),
        "inconsistent_alternative_pairwise_scenarios": (
            inconsistent
        ),
        "checks": checks,
        "upstream_status": evidence,
        "artifact_inventory": inventory,
        "limitations": [
            "Four adapted published comparisons and "
            "two unsourced scenario assumptions.",
            "No genuine Kaduwela expert AHP yet.",
            "Cross-study weight transfer is not "
            "independently validated.",
            "Relative physical/accessibility "
            "suitability only.",
            "No legal development permission implied.",
            "No independent suitability accuracy claimed.",
            "Categorical class assignments show "
            "sensitivity to methodology choices.",
            "Expert assessment or approved methodology "
            "revision remains necessary.",
        ],
    }

    report_json = (
        REPORT / "03i_provisional_release_audit.json"
    )

    report_md = (
        REPORT / "03i_viva_summary_provisional.md"
    )

    report_json.write_text(
        json.dumps(results, indent=2),
        encoding="utf-8",
    )

    # -----------------------------------------------
    # 7. Simple viva-ready summary
    # -----------------------------------------------

    lines = [
        "# Development Suitability - Provisional Research Summary",
        "",
        "**Research:** J26-DS-338",
        "**Study area:** Kaduwela Municipal Council",
        "",
        "**Status:** Technical QA PASS; "
        "approved expert AHP NOT completed.",
        "",
        f"**Coverage:** 9,240 grid cells; "
        f"{kmc_total / 1e6:.6f} km2.",
        "",
        "## Methodology",
        "",
        "GIS-MCDA weighted sum using four P05-P95 "
        "standardized criteria: road proximity, "
        "500 m road density, 500 m building context, "
        "and mean slope.",
        "",
        "## Provisional AHP weights",
        "",
        *[
            f"- {name}: {weight * 100:.2f}%"
            for name, weight in zip(names, weights)
        ],
        f"- CR: {cr:.6f} (< 0.10 mathematically).",
        "",
        f"**Area-weighted suitability score:** "
        f"{np.average(observed, weights=areas):.4f}",
        "",
        f"**High + Very High area:** "
        f"{high_pct:.2f}% of KMC.",
        "",
        "## Class distribution",
        "",
        "| Class | Cells | Area (km2) | KMC area (%) |",
        "|---|---:|---:|---:|",
        *[
            f"| {row['class']} | {row['count']:,} | "
            f"{row['area_km2']:.4f} | "
            f"{row['kmc_area_pct']:.2f} |"
            for row in class_area
        ],
        "",
        "## Sensitivity",
        "",
        f"19 scenarios; minimum Spearman correlation "
        f"{results['minimum_sensitivity_spearman']:.4f}; "
        f"maximum changed-class area "
        f"{results['maximum_changed_kmc_area_pct']:.2f}%.",
        "",
        "Continuous rankings were comparatively stable, "
        "but suitability classes were more sensitive "
        "to modelling decisions.",
        "",
        "## Important limitations",
        "",
        "The AHP matrix contains four cross-study adapted "
        "comparisons and two researcher scenario assumptions.",
        "",
        "This is not a genuine Kaduwela expert assessment "
        "and does not yet meet the approved expert-AHP requirement.",
        "",
        "The map does not establish legal development "
        "permission, land availability or independent "
        "development-suitability accuracy.",
        "",
        "All results remain PROVISIONAL pending "
        "expert review or an approved methodology change.",
    ]

    report_md.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    print(
        "Checks passed:",
        sum(checks.values()),
        "/",
        len(checks),
    )

    print("Audit JSON:", report_json)
    print("Viva summary:", report_md)

    print("\n" + "=" * 64)
    print("STAGE 3.11: PASS_PROVISIONAL")
    print("Computational pipeline: AUDITED")
    print("Approved expert AHP: STILL PENDING")
    print("=" * 64)


if __name__ == "__main__":
    main()
