
"""
J26-DS-338 | Section 03 | Stage 3.9
Provisional Development Suitability Classification and Mapping

Method:
- Five equal-interval classes on the 0-1 MCDA score.
- Preserve continuous scores and weighted contributions.
- Use actual Grid-intersection-KMC geometries and areas.
- Generate CSV, GeoPackage, PNG map and one QA JSON.

IMPORTANT:
- This is a provisional literature-informed AHP scenario.
- Classes are relative index categories, not legal zoning.
- No retraining, reweighting or rescoring.
"""

from pathlib import Path
import json

import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyogrio
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch


ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = (
    ROOT / "data/processed/development_suitability/"
    "weighted_suitability"
)

REPORT_DIR = (
    ROOT / "outputs/reports/development_suitability"
)

MAP_DIR = (
    ROOT / "outputs/maps/development_suitability"
)

CSV_IN = (
    DATA_DIR / "03f_provisional_weighted_suitability_100m.csv"
)

GPKG_IN = (
    DATA_DIR / "03f_provisional_weighted_suitability_100m.gpkg"
)

MCDA_QA = (
    REPORT_DIR / "03f_provisional_weighted_suitability_qa.json"
)

AHP_QA = (
    REPORT_DIR / "03e_provisional_ahp_audit.json"
)

BOUNDARY = (
    ROOT / "data/processed/boundaries/kmc/"
    "kaduwela_kmc_boundary_utm44n.gpkg"
)

SCORE = "provisional_suitability_score"

CLASS_NAMES = [
    "VERY_LOW",
    "LOW",
    "MODERATE",
    "HIGH",
    "VERY_HIGH",
]

CLASS_LIMITS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

CLASS_COLORS = [
    "#B54849",
    "#E99D5A",
    "#F2D47A",
    "#A9C98B",
    "#378E69",
]


def main():

    print("=" * 65)
    print("STAGE 3.9 - PROVISIONAL SUITABILITY MAPPING")
    print("=" * 65)

    for path in [
        CSV_IN, GPKG_IN, MCDA_QA, AHP_QA, BOUNDARY
    ]:
        if not path.is_file():
            raise FileNotFoundError(path)

    mcda_audit = json.loads(
        MCDA_QA.read_text(encoding="utf-8")
    )

    ahp_audit = json.loads(
        AHP_QA.read_text(encoding="utf-8")
    )

    if mcda_audit.get("status") != "PASS_PROVISIONAL":
        raise ValueError("Stage 3.8 has not passed")

    if ahp_audit.get("is_approved_final_ahp") is not False:
        raise ValueError("Unexpected AHP provenance")

    # -------------------------------------------------
    # 1. Read and verify continuous scores
    # -------------------------------------------------

    print("\n[1] INPUT VALIDATION")

    df = pd.read_csv(CSV_IN)
    source_gis = gpd.read_file(GPKG_IN)
    kmc = gpd.read_file(BOUNDARY)

    if (
        len(df) != 9240
        or len(source_gis) != 9240
        or len(kmc) != 1
    ):
        raise ValueError("Unexpected GIS feature count")

    if (
        not df.grid_id.is_unique
        or not source_gis.grid_id.is_unique
    ):
        raise ValueError("Duplicate grid IDs")

    if (
        df.grid_id.isna().any()
        or source_gis.grid_id.isna().any()
    ):
        raise ValueError("Missing grid IDs")

    if (
        source_gis.crs is None
        or source_gis.crs.to_epsg() != 32644
        or kmc.crs is None
        or kmc.crs.to_epsg() != 32644
    ):
        raise ValueError("Incorrect GIS coordinate system")

    if (
        source_gis.geometry.isna().any()
        or not source_gis.geometry.is_valid.all()
    ):
        raise ValueError("Invalid source geometries")

    scores = df[SCORE].to_numpy(dtype=float)

    if not np.isfinite(scores).all():
        raise ValueError("Non-finite suitability scores")

    if np.any((scores < 0) | (scores > 1)):
        raise ValueError("Suitability scores outside 0-1")

    # Confirm CSV and GPKG contain identical scores.
    comparison = df[["grid_id", SCORE]].merge(
        source_gis[["grid_id", SCORE]],
        on="grid_id",
        validate="one_to_one",
        suffixes=("_csv", "_gis"),
    )

    score_difference = float(
        np.max(
            np.abs(
                comparison[f"{SCORE}_csv"].to_numpy()
                - comparison[f"{SCORE}_gis"].to_numpy()
            )
        )
    )

    if (
        len(comparison) != 9240
        or score_difference > 1e-9
    ):
        raise ValueError("CSV/GIS score mismatch")

    print("Grid cells:", len(df))
    print("Score range:", round(scores.min(), 6),
          "to", round(scores.max(), 6))
    print("CSV/GIS maximum score difference:", score_difference)

    # -------------------------------------------------
    # 2. Five-class equal-interval classification
    # -------------------------------------------------

    print("\n[2] CLASSIFY SUITABILITY")

    # Exactly 0.2 belongs to LOW; exactly 0.4 to MODERATE.
    # The final class includes 1.0.
    class_indexes = np.digitize(
        scores,
        bins=CLASS_LIMITS[1:-1],
        right=False,
    )

    df["suitability_class_code"] = class_indexes + 1

    df["suitability_class"] = [
        CLASS_NAMES[i] for i in class_indexes
    ]

    df["classification_method"] = "EQUAL_INTERVAL_5"

    # -------------------------------------------------
    # 3. Attach existing GIS geometry
    # -------------------------------------------------

    geo = source_gis[["grid_id", "geometry"]].merge(
        df,
        on="grid_id",
        how="inner",
        validate="one_to_one",
    )

    if len(geo) != 9240:
        raise ValueError("Geometry join lost grid cells")

    actual_area = geo.geometry.area.to_numpy(dtype=float)

    stored_area = geo[
        "kmc_overlap_area_m2"
    ].to_numpy(dtype=float)

    area_error = float(
        np.max(np.abs(actual_area - stored_area))
    )

    kmc_area = float(kmc.geometry.area.sum())

    study_area_difference = abs(
        float(stored_area.sum()) - kmc_area
    )

    # -------------------------------------------------
    # 4. Class statistics using actual KMC area
    # -------------------------------------------------

    print("\n[3] CLASS DISTRIBUTION")

    summary = []

    for i, name in enumerate(CLASS_NAMES):

        subset = df[
            df["suitability_class_code"] == i + 1
        ]

        count = len(subset)

        area_m2 = float(
            subset["kmc_overlap_area_m2"].sum()
        )

        area_km2 = area_m2 / 1e6

        area_pct = (
            100.0 * area_m2 / kmc_area
        )

        cell_pct = (
            100.0 * count / len(df)
        )

        record = {
            "class": name,
            "class_code": i + 1,
            "lower_score": CLASS_LIMITS[i],
            "upper_score": CLASS_LIMITS[i + 1],
            "grid_count": count,
            "grid_percentage": cell_pct,
            "kmc_area_km2": area_km2,
            "kmc_area_percentage": area_pct,
            "mean_score": (
                float(subset[SCORE].mean())
                if count > 0 else None
            ),
        }

        summary.append(record)

        print(
            f"{name:10s} | "
            f"Cells: {count:5d} | "
            f"Area: {area_km2:9.4f} km2 | "
            f"KMC: {area_pct:6.2f}%"
        )

    # -------------------------------------------------
    # 5. Research validation
    # -------------------------------------------------

    print("\n[4] VALIDATION")

    checks = {
        "9240_cells_preserved": len(df) == len(geo) == 9240,
        "unique_grid_ids": df.grid_id.is_unique,
        "all_scores_finite": bool(np.isfinite(scores).all()),
        "scores_in_0_1": bool(
            ((scores >= 0) & (scores <= 1)).all()
        ),
        "five_defined_classes": len(CLASS_NAMES) == 5,
        "every_cell_classified": bool(
            df["suitability_class"].notna().all()
        ),
        "class_codes_valid": bool(
            df["suitability_class_code"].between(1, 5).all()
        ),
        "classification_reproduced": bool(
            np.array_equal(
                class_indexes,
                np.digitize(
                    scores,
                    bins=CLASS_LIMITS[1:-1],
                    right=False,
                ),
            )
        ),
        "all_class_counts_sum_9240": (
            sum(x["grid_count"] for x in summary) == 9240
        ),
        "class_area_reconciles_kmc": (
            study_area_difference <= 1.0
        ),
        "cell_geometry_area_reconciled": (
            area_error <= 1.0
        ),
        "csv_gis_scores_identical": (
            score_difference <= 1e-9
        ),
    }

    for name, passed in checks.items():
        print(
            f"[{'PASS' if passed else 'FAIL'}] {name}"
        )

    if not all(checks.values()):
        raise SystemExit(
            "Stage 3.9 QA failed. No outputs exported."
        )

    # -------------------------------------------------
    # 6. Export classified data
    # -------------------------------------------------

    print("\n[5] EXPORT CLASSIFIED GIS DATA")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    MAP_DIR.mkdir(parents=True, exist_ok=True)

    csv_out = (
        DATA_DIR /
        "03g_provisional_suitability_classes_100m.csv"
    )

    gpkg_out = (
        DATA_DIR /
        "03g_provisional_suitability_classes_100m.gpkg"
    )

    map_out = (
        MAP_DIR /
        "03g_provisional_suitability_map.png"
    )

    qa_out = (
        REPORT_DIR /
        "03g_suitability_classification_qa.json"
    )

    df.to_csv(csv_out, index=False)

    geo.to_file(
        gpkg_out,
        layer="provisional_suitability_classes",
        driver="GPKG",
    )

    # -------------------------------------------------
    # 7. Create actual GIS map
    # -------------------------------------------------

    print("\n[6] GENERATE GIS MAP")

    cmap = ListedColormap(CLASS_COLORS)

    norm = BoundaryNorm(
        np.arange(-0.5, 5.5, 1),
        cmap.N,
    )

    fig, ax = plt.subplots(figsize=(12, 9))

    geo["map_class_index"] = class_indexes

    geo.plot(
        column="map_class_index",
        cmap=cmap,
        norm=norm,
        ax=ax,
        linewidth=0,
        legend=False,
    )

    kmc.boundary.plot(
        ax=ax,
        color="#303030",
        linewidth=0.75,
    )

    ax.set_aspect("equal")
    ax.set_axis_off()

    xmin, ymin, xmax, ymax = geo.total_bounds

    # Simple north indication.
    ax.annotate(
        "N",
        xy=(0.96, 0.97),
        xytext=(0.96, 0.87),
        xycoords="axes fraction",
        textcoords="axes fraction",
        ha="center",
        va="center",
        fontsize=12,
        fontweight="bold",
        arrowprops={
            "arrowstyle": "-|>",
            "color": "#303030",
            "linewidth": 1.5,
        },
    )

    # EPSG:32644 uses metres, enabling a metric scale bar.
    scale_x = xmin + (xmax - xmin) * 0.08
    scale_y = ymin + (ymax - ymin) * 0.06

    ax.plot(
        [scale_x, scale_x + 2000],
        [scale_y, scale_y],
        color="#252525",
        linewidth=3,
    )

    ax.text(
        scale_x + 1000,
        scale_y + (ymax - ymin) * 0.018,
        "2 km",
        ha="center",
        fontsize=9,
        color="#252525",
    )

    legend_handles = [
        Patch(
            facecolor=CLASS_COLORS[i],
            label=(
                f"{name.replace('_', ' ').title()}"
                f"  ({CLASS_LIMITS[i]:.1f}-"
                f"{CLASS_LIMITS[i+1]:.1f})"
            ),
        )
        for i, name in enumerate(CLASS_NAMES)
    ]

    fig.legend(
        handles=legend_handles,
        title="Relative suitability",
        loc="center right",
        bbox_to_anchor=(0.99, 0.55),
        frameon=False,
        fontsize=10,
    )

    fig.suptitle(
        "Kaduwela Municipal Council\n"
        "Provisional Development Suitability",
        fontsize=17,
        fontweight="bold",
        y=0.96,
    )

    fig.text(
        0.5, 0.075,
        "100 m grid | Equal-interval classification | "
        "Literature-informed provisional AHP",
        ha="center",
        fontsize=10,
    )

    fig.text(
        0.5, 0.045,
        "Research scenario only - not authoritative zoning "
        "or legal development permission",
        ha="center",
        fontsize=9,
        color="#A04141",
    )

    fig.subplots_adjust(
        left=0.03,
        right=0.79,
        top=0.87,
        bottom=0.11,
    )

    fig.savefig(
        map_out,
        dpi=220,
        facecolor="white",
    )

    plt.close(fig)

    # -------------------------------------------------
    # 8. Final export checks
    # -------------------------------------------------

    saved_csv = pd.read_csv(csv_out)

    saved_gis = pyogrio.read_info(gpkg_out)

    checks["csv_export_count"] = (
        len(saved_csv) == 9240
        and saved_csv.grid_id.is_unique
    )

    checks["gis_export_count"] = (
        saved_gis["features"] == 9240
    )

    checks["map_export_exists"] = (
        map_out.is_file()
        and map_out.stat().st_size > 10000
    )

    status = (
        "PASS_PROVISIONAL"
        if all(checks.values())
        else "REVIEW_REQUIRED"
    )

    report = {
        "stage": "3.9",
        "status": status,
        "method": "five_class_equal_interval",
        "class_limits": CLASS_LIMITS,
        "grid_cells": len(df),
        "kmc_area_km2": kmc_area / 1e6,
        "study_area_difference_m2": (
            study_area_difference
        ),
        "maximum_cell_area_error_m2": area_error,
        "class_distribution": summary,
        "checks": checks,
        "weight_provenance": (
            "PROVISIONAL_LITERATURE_ADAPTED"
        ),
        "is_final_approved_suitability": False,
        "limitations": [
            "Classes are relative analytical categories.",
            "Equal intervals are not engineering thresholds.",
            "A high score does not imply vacant land.",
            "Provisional UDA zoning is not used.",
            "Expert-derived final AHP weights are pending.",
            "Classification and weight sensitivity are pending.",
            "No independent suitability accuracy is claimed.",
        ],
    }

    qa_out.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print("\n[7] OUTPUTS")

    print("CSV:", csv_out)
    print("GPKG:", gpkg_out)
    print("Map:", map_out)
    print("QA:", qa_out)

    print("\n" + "=" * 65)
    print("STAGE 3.9:", status)
    print("Total classified cells:", len(df))
    print("Classification: Five equal-interval classes")
    print("Final expert approval: PENDING")
    print("=" * 65)

    if status != "PASS_PROVISIONAL":
        raise SystemExit(
            "Stage 3.9 export validation failed."
        )


if __name__ == "__main__":
    main()
