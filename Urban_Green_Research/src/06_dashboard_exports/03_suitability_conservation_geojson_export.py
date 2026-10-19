
"""
J26-DS-338 | Urban Green Research

BATCH A - DEVELOPMENT SUITABILITY + CONSERVATION GIS

Export two scientifically verified WGS84 GeoJSON maps.

- Suitability: final continuous expert-AHP GIS-MCDA score
- Conservation: final frozen GI x frozen UP priority

Read-only research sources. No model recalculation,
new criteria weighting, priority classification,
or modification of other components.
"""

from pathlib import Path
from collections import Counter
import csv
import hashlib
import json
import math
import tempfile

import geopandas as gpd


ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "outputs" / "reports"
DASHBOARD = ROOT / "outputs" / "dashboard"

SUIT_RELEASE = REPORTS / (
    "16g_01_final_expert_ahp_suitability_release.json"
)
CONSERVATION_QA = (
    REPORTS / "conservation_intelligence"
    / "17o_conservation_final_qa.json"
)

SUIT_OUTPUT = DASHBOARD / "development_suitability"
CONS_OUTPUT = DASHBOARD / "conservation_intelligence"

SUIT_FIELDS = [
    "grid_id",
    "kmc_overlap_area_m2",
    "kmc_overlap_fraction",
    "boundary_cell",
    "coverage_group",
    "terrain_fallback_applied",
    "c1_major_road_distance_m",
    "c2_road_density_500m_km_per_km2",
    "c4_slope_mean_deg",
    "c1_road_access_score",
    "c2_road_context_score",
    "c4_slope_score",
    "c1_expert_ahp_weight",
    "c2_expert_ahp_weight",
    "c4_expert_ahp_weight",
    "development_suitability_score",
    "suitability_class",
]

CONS_FIELDS = [
    "grid_id",
    "kmc_overlap_area_m2",
    "current_green_condition_status",
    "green_importance_status",
    "green_importance_frozen",
    "G1_current_green_0_1",
    "G2_historical_persistence_0_1",
    "G3_local_green_context_0_1",
    "green_importance_index_0_1",
    "future_pressure_domain_status",
    "future_pressure_origin_year",
    "future_pressure_period",
    "future_urbanization_pressure_score",
    "future_threshold_flag",
    "historical_g2b_robust_exposure_share_250m",
    "primary_priority_status",
    "primary_conservation_priority_0_1",
    "primary_priority_rank_common_domain",
    "primary_priority_percentile_common_domain",
    "priority_classes_assigned",
]

EXPECTED_CONS_STATUS = {
    "FUTURE_CONSERVATION_PRIORITY_CALCULATED": 4609,
    "NOT_IN_PRESSURE_DOMAIN": 4118,
    "GI_UNSCORED_PRESSURE_AVAILABLE": 11,
    "GI_UNSCORED_AND_PRESSURE_NOT_IN_DOMAIN": 502,
}


def read_json(path):
    with path.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def read_csv(path):
    with path.open(
        "r", encoding="utf-8-sig", newline=""
    ) as f:
        return list(csv.DictReader(f))


def digest(path):
    h = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(
            lambda: f.read(1024 * 1024), b""
        ):
            h.update(chunk)

    return h.hexdigest()


def numeric(value):
    if value is None or value == "":
        return None

    result = float(value)

    if not math.isfinite(result):
        raise ValueError("Non-finite numeric value")

    return result



def same(a, b, tolerance=1e-8):
    """
    Compare numeric research values safely.

    CSV empty values and GIS NaN values both
    represent missing data.

    Do not replace missing values with zero.
    """

    def normalize(value):
        if value is None or value == "":
            return None

        result = float(value)

        if math.isnan(result):
            return None

        if not math.isfinite(result):
            raise ValueError(
                "Unexpected infinite numeric value"
            )

        return result

    a = normalize(a)
    b = normalize(b)

    if a is None or b is None:
        return a is None and b is None

    return math.isclose(
        a,
        b,
        rel_tol=0,
        abs_tol=tolerance,
    )



def load_gis(path, layer, fields):
    gdf = gpd.read_file(path, layer=layer)

    assert len(gdf) == 9240
    assert gdf.crs.to_epsg() == 32644
    assert gdf["grid_id"].nunique() == 9240
    assert gdf.geometry.notna().all()
    assert (~gdf.geometry.is_empty).all()
    assert gdf.geometry.is_valid.all()

    missing = set(fields) - set(gdf.columns)

    if missing:
        raise ValueError(
            f"Missing GIS columns: {sorted(missing)}"
        )

    return gdf


def verify_suitability():
    print("\n========== 1. FINAL SUITABILITY ==========")

    release = read_json(SUIT_RELEASE)

    assert release["status"] == (
        "PASS_SECTION03_FINAL_CONTINUOUS_"
        "SUITABILITY_WITH_DOCUMENTED_LIMITATIONS"
    )

    scope = release["model_scope"]
    assert scope["continuous_suitability_completed"] is True
    assert scope["final_policy_suitability_classes"] is False

    weights = {
        key: float(value["weight"])
        for key, value in release["criteria"].items()
    }

    assert set(weights) == {"C1", "C2", "C4"}
    assert same(weights["C1"], 0.20)
    assert same(weights["C2"], 0.60)
    assert same(weights["C4"], 0.20)

    assert same(
        release["expert_consultation"]["consistency_ratio"],
        0.0,
    )

    files = release["final_deliverables"]

    source_csv = ROOT / files["suitability_csv"]
    source_gpkg = ROOT / files["suitability_gpkg"]

    # Verify the original frozen research data.
    for path in (source_csv, source_gpkg):
        expected = release["input_sha256"][
            path.relative_to(ROOT).as_posix()
        ]
        assert digest(path) == expected, path

    rows = read_csv(source_csv)
    assert len(rows) == 9240

    lookup = {r["grid_id"]: r for r in rows}
    assert len(lookup) == 9240

    gdf = load_gis(
        source_gpkg,
        files["suitability_gpkg_layer"],
        SUIT_FIELDS,
    )

    assert set(gdf["grid_id"]) == set(lookup)

    for row in gdf.itertuples(index=False):
        original = lookup[row.grid_id]

        actual = numeric(
            original["development_suitability_score"]
        )

        computed = (
            0.20 * numeric(original["c1_road_access_score"])
            + 0.60 * numeric(original["c2_road_context_score"])
            + 0.20 * numeric(original["c4_slope_score"])
        )

        assert same(actual, computed)
        assert same(actual, row.development_suitability_score)
        assert 0 <= actual <= 1

        assert original["suitability_class"] == "NOT_YET_CLASSIFIED"
        assert row.suitability_class == "NOT_YET_CLASSIFIED"

    print("[PASS] Frozen files match release SHA256")
    print("[PASS] 9240 original AHP scores")
    print("[PASS] C1/C2/C4 weighted formula")
    print("[PASS] No artificial suitability classes")

    metadata = {
        "status": "ok",
        "research_component": "Development Suitability",
        "release_status": release["status"],
        "method": release["method"],
        "formula": release["formula"],
        "criteria": release["criteria"],
        "expert_consultation": release["expert_consultation"],
        "grid_count": 9240,
        "score_field": "development_suitability_score",
        "score_range": [0, 1],
        "continuous_score": True,
        "final_policy_classes_assigned": False,
        "distribution": release["suitability_distribution"],
        "sensitivity": release["sensitivity"],
        "limitations": release["limitations"],
        "source_stage": "16G",
        "source_gis": files["suitability_gpkg"],
        "source_gis_layer": files["suitability_gpkg_layer"],
        "crs": "EPSG:4326",
        "source_crs": "EPSG:32644",
        "geojson_file": "suitability_map.geojson",
        "join_field": "grid_id",
    }

    return gdf[SUIT_FIELDS + ["geometry"]].copy(), metadata


def verify_conservation():
    print("\n========== 2. FINAL CONSERVATION ==========")

    qa = read_json(CONSERVATION_QA)

    assert qa["status"] == (
        "PASS_FINAL_GIS_PACKAGE_WITH_DOCUMENTED_LIMITATIONS"
    )

    scope = qa["scope"]

    assert scope["all_KMC_grid_cells"] == 9240
    assert scope["GI_scored_cells"] == 8727
    assert scope["primary_CP_scored_common_domain"] == 4609
    assert scope["unscored_primary_CP_nulls"] == 4631

    source_gpkg = ROOT / qa["GIS_package"]["path"]

    assert digest(source_gpkg) == (
        qa["GIS_package"]["sha256"]
    )

    source_csv = (
        ROOT / "data" / "processed"
        / "conservation_intelligence"
        / "conservation_priority"
        / "17m_primary_conservation_priority_100m.csv"
    )

    rows = read_csv(source_csv)

    assert len(rows) == 9240

    lookup = {r["grid_id"]: r for r in rows}
    assert len(lookup) == 9240

    gdf = load_gis(
        source_gpkg,
        "primary_conservation_priority",
        CONS_FIELDS,
    )

    assert set(gdf["grid_id"]) == set(lookup)

    statuses = Counter()
    cp_count = 0
    gi_count = 0

    for row in gdf.itertuples(index=False):
        original = lookup[row.grid_id]

        status = row.primary_priority_status
        statuses[status] += 1

        assert status == original["primary_priority_status"]

        gi = numeric(original["green_importance_index_0_1"])
        up = numeric(original["future_urbanization_pressure_score"])
        cp = numeric(original["primary_conservation_priority_0_1"])

        assert same(gi, row.green_importance_index_0_1)
        assert same(up, row.future_urbanization_pressure_score)
        assert same(cp, row.primary_conservation_priority_0_1)

        if gi is not None:
            gi_count += 1
            assert 0 <= gi <= 1

        if cp is not None:
            cp_count += 1

            assert gi is not None and up is not None
            assert 0 <= up <= 1
            assert 0 <= cp <= 1
            assert same(cp, gi * up)
            assert status == (
                "FUTURE_CONSERVATION_PRIORITY_CALCULATED"
            )
        
        else:
            assert same(
                None,
                row.primary_conservation_priority_0_1,
            )
            assert same(
                None,
                row.primary_priority_rank_common_domain,
            )
            assert same(
                None,
                row.primary_priority_percentile_common_domain,
            )


        assert original["priority_classes_assigned"] == "False"

    assert dict(statuses) == EXPECTED_CONS_STATUS
    assert gi_count == 8727
    assert cp_count == 4609

    print("[PASS] Final conservation GIS SHA256")
    print("[PASS] 9240 aligned original grid IDs")
    print("[PASS] 8727 Green Importance scores")
    print("[PASS] 4609 valid conservation priority scores")
    print("[PASS] 4631 missing priority scores preserved")
    print("[PASS] CP = GI x UP")
    print("[PASS] No artificial conservation classes")

    metadata = {
        "status": "ok",
        "research_component": "Conservation Intelligence",
        "release_status": qa["status"],
        "methodology": qa["methodology"],
        "grid_count": 9240,
        "green_importance_scored_count": 8727,
        "primary_priority_scored_count": 4609,
        "primary_priority_unscored_count": 4631,
        "status_counts": EXPECTED_CONS_STATUS,
        "score_field": "primary_conservation_priority_0_1",
        "importance_field": "green_importance_index_0_1",
        "score_range": [0, 1],
        "final_priority_classes_assigned": False,
        "distribution": qa["primary_CP_distribution"],
        "sensitivity": qa["sensitivity"],
        "source_stage": "17O",
        "source_gis": qa["GIS_package"]["path"],
        "source_gis_layer": "primary_conservation_priority",
        "crs": "EPSG:4326",
        "source_crs": "EPSG:32644",
        "geojson_file": "conservation_map.geojson",
        "join_field": "grid_id",
        "interpretation": (
            "Relative conservation priority for the "
            "4609 cells with both Green Importance and "
            "model-derived future pressure. "
            "Not a calibrated probability or field-validated "
            "ecological risk assessment."
        ),
    }

    return gdf[CONS_FIELDS + ["geometry"]].copy(), metadata


def write_map(gdf, metadata, folder):
    transformed = gdf.to_crs(epsg=4326)

    assert transformed.crs.to_epsg() == 4326
    assert transformed.geometry.is_valid.all()

    collection = json.loads(
        transformed.to_json(
            na="null",
            drop_id=True,
        )
    )

    assert len(collection["features"]) == 9240

    metadata["bounds_west_south_east_north"] = (
        transformed.total_bounds.tolist()
    )

    folder.mkdir()

    map_path = folder / metadata["geojson_file"]

    map_path.write_text(
        json.dumps(
            collection,
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        ),
        encoding="utf-8",
    )

    (folder / "metadata.json").write_text(
        json.dumps(
            metadata,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        ),
        encoding="utf-8",
    )

    # Verify actual files before release.
    saved = read_json(map_path)
    assert len(saved["features"]) == 9240
    assert len({
        f["properties"]["grid_id"]
        for f in saved["features"]
    }) == 9240

    print(
        "[PASS]",
        map_path.name,
        "- 9240 features,",
        map_path.stat().st_size,
        "bytes",
    )


def main():
    print("\n========== COMBINED GIS EXPORT ==========")

    if SUIT_OUTPUT.exists() or CONS_OUTPUT.exists():
        raise FileExistsError(
            "One or both output folders already exist. "
            "Refusing to overwrite."
        )

    suitability, suitability_meta = verify_suitability()
    conservation, conservation_meta = verify_conservation()

    DASHBOARD.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(
        prefix=".green_gis_export_",
        dir=DASHBOARD,
    ) as temporary:
        tmp = Path(temporary)

        suit_tmp = tmp / "development_suitability"
        cons_tmp = tmp / "conservation_intelligence"

        print("\n========== 3. WGS84 GEOJSON EXPORT ==========")

        write_map(suitability, suitability_meta, suit_tmp)
        write_map(conservation, conservation_meta, cons_tmp)

        # Release only after both maps pass verification.
        suit_tmp.rename(SUIT_OUTPUT)
        cons_tmp.rename(CONS_OUTPUT)

    print("\n========== FINAL RESULT ==========")
    print("[PASS] Development Suitability GIS export")
    print("[PASS] Conservation Intelligence GIS export")
    print("[PASS] Frozen scientific sources unchanged")
    print("[PASS] No retraining or recalculation")
    print("Suitability:", SUIT_OUTPUT)
    print("Conservation:", CONS_OUTPUT)


if __name__ == "__main__":
    main()
