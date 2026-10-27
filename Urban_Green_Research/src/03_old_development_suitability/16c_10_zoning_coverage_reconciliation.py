
"""
STEP 16C-J - ZONING COVERAGE RECONCILIATION

Component:
Urban Green Transition, Urbanization Pressure
and Conservation Intelligence

Purpose:
1. Resolve the two Step 16C-I area discrepancies.
2. Separate classified, unclassified and unmapped areas.
3. Verify the coverage directly using polygon geometry.
4. Compare all results with supplied UDA QA metadata.
5. Save reproducible reconciliation and QA reports.

Important:
- The supplied zoning GIS is provisional and unverified.
- Unclassified is different from truly unmapped.
- Ambiguous-name areas overlap other zoning categories.
- No suitability scoring, AHP or grid overlay.
- Previous research data and reports are not modified.
"""

from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json

import geopandas as gpd
import numpy as np
import pandas as pd


# ============================================================
# 1. PROJECT PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = (
    ROOT / "data/raw/uda/zoning_provisional"
    / "provided_package_extracted/New data"
)

REPORTS = ROOT / "outputs/reports"

RAW_INPUT = (
    ROOT / "data/processed/development_suitability"
    / "raw_criteria"
    / "16b_raw_suitability_criteria_100m.csv"
)

MAIN_FILE = (
    DATA_DIR
    / "Kaduwela_UDA_2024_AUDITED_PROVISIONAL.gpkg"
)

SCREEN_FILE = (
    DATA_DIR
    / "Kaduwela_Conservation_Review_Screening_PROVISIONAL.gpkg"
)

BOUNDARY_FILE = (
    DATA_DIR / "Kaduwela_MC_Boundary.gpkg"
)

SUPPLIED_QA = DATA_DIR / "QA_METRICS.json"

PREVIOUS_H = (
    REPORTS / "16c_h_zoning_inventory_summary.json"
)

PREVIOUS_I = (
    REPORTS / "16c_i_zoning_coverage_summary.json"
)

SOURCE_INVENTORY = (
    REPORTS / "16c_h_source_inventory.csv"
)

COVERAGE_CSV = (
    REPORTS / "16c_j_reconciled_zoning_coverage.csv"
)

CROSSCHECK_CSV = (
    REPORTS / "16c_j_reconciled_qa_crosscheck.csv"
)

QA_CSV = (
    REPORTS / "16c_j_zoning_reconciliation_qa.csv"
)

SUMMARY_JSON = (
    REPORTS / "16c_j_zoning_reconciliation_summary.json"
)

EPSG = 32644

AREA_TOLERANCE_HA = 0.05
PERCENT_TOLERANCE = 0.01


# ============================================================
# 2. HELPERS
# ============================================================

def sha256(path):
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(
            lambda: file.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def read_json(path):
    with path.open(
        encoding="utf-8-sig"
    ) as file:
        return json.load(file)


def union_all(geometries):
    if hasattr(geometries, "union_all"):
        return geometries.union_all()

    return geometries.unary_union


def hectares(geometry):
    return float(geometry.area / 10000.0)


def percent(value, total):
    return float(100 * value / total)


def close_to(a, b, tolerance):
    return abs(float(a) - float(b)) <= tolerance


# ============================================================
# 3. START
# ============================================================

print("=" * 95)
print("STEP 16C-J - ZONING COVERAGE RECONCILIATION")
print("=" * 95)

required_files = [
    RAW_INPUT,
    MAIN_FILE,
    SCREEN_FILE,
    BOUNDARY_FILE,
    SUPPLIED_QA,
    PREVIOUS_H,
    PREVIOUS_I,
    SOURCE_INVENTORY,
]

for path in required_files:
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing required file: {path}"
        )

REPORTS.mkdir(
    parents=True,
    exist_ok=True,
)

previous_h = read_json(PREVIOUS_H)
previous_i = read_json(PREVIOUS_I)
supplied = read_json(SUPPLIED_QA)

inventory = pd.read_csv(
    SOURCE_INVENTORY
)

raw_hash = sha256(RAW_INPUT)


# ============================================================
# 4. INITIAL QA
# ============================================================

checks = []


def check(name, passed, detail):
    checks.append({
        "check": name,
        "passed": bool(passed),
        "detail": str(detail),
    })


check(
    "Previous Step 16C-H passed",
    previous_h.get("qa_status") == "PASS",
    previous_h.get("qa_status"),
)

check(
    "Previous Step 16C-I technical QA passed",
    previous_i.get("qa_status") == "PASS",
    previous_i.get("qa_status"),
)

check(
    "Frozen 16B input matches previous audits",
    raw_hash == previous_h.get("frozen_input_sha256")
    and raw_hash == previous_i.get("frozen_raw_sha256"),
    "SHA256 checked",
)

check(
    "Previous audit recorded two discrepancies",
    previous_i.get("crosscheck_review_count") == 2,
    "Both discrepancies will be reviewed",
)


# ============================================================
# 5. VERIFY GIS SOURCE PROVENANCE
# ============================================================

source_paths = [
    MAIN_FILE,
    SCREEN_FILE,
    BOUNDARY_FILE,
]

source_hashes = {}

for path in source_paths:

    current_hash = sha256(path)
    source_hashes[path.name] = current_hash

    rows = inventory[
        inventory["file"] == path.name
    ]

    matched = (
        len(rows) == 1
        and rows.iloc[0]["sha256"] == current_hash
    )

    check(
        f"GIS source unchanged: {path.name}",
        matched,
        "Matches Step 16C-H source inventory",
    )


# ============================================================
# 6. LOAD GEOMETRY
# ============================================================

main_gdf = gpd.read_file(
    MAIN_FILE,
    layer="UDA_2024_ZONE_ESTIMATES",
)

unknown_gdf = gpd.read_file(
    SCREEN_FILE,
    layer="UNKNOWN_ZONING_130",
)

ambiguous_gdf = gpd.read_file(
    MAIN_FILE,
    layer="QA_171_Ambiguous_Names",
)

boundary_gdf = gpd.read_file(
    BOUNDARY_FILE,
    layer="Kaduwela_MC_Boundary",
)

layers = {
    "classified": main_gdf,
    "unclassified": unknown_gdf,
    "ambiguous": ambiguous_gdf,
    "boundary": boundary_gdf,
}

expected_counts = {
    "classified": 508,
    "unclassified": 130,
    "ambiguous": 171,
    "boundary": 1,
}

for name, gdf in layers.items():

    geometry_valid = bool(
        gdf.crs is not None
        and gdf.geometry.notna().all()
        and (~gdf.geometry.is_empty).all()
        and gdf.geometry.is_valid.all()
    )

    check(
        f"{name} geometry and CRS valid",
        geometry_valid,
        f"Features: {len(gdf)}",
    )

    check(
        f"{name} feature count correct",
        len(gdf) == expected_counts[name],
        f"Expected: {expected_counts[name]}",
    )

if not all(row["passed"] for row in checks):
    pd.DataFrame(checks).to_csv(
        QA_CSV,
        index=False,
    )
    raise RuntimeError(
        "Initial QA failed. Check the QA report."
    )


# ============================================================
# 7. PROJECT ALL GEOMETRIES
# ============================================================

projected = {
    name: gdf.to_crs(epsg=EPSG)
    for name, gdf in layers.items()
}

boundary = union_all(
    projected["boundary"].geometry
)

classified = union_all(
    projected["classified"].geometry
).intersection(boundary)

unknown = union_all(
    projected["unclassified"].geometry
).intersection(boundary)

ambiguous = union_all(
    projected["ambiguous"].geometry
).intersection(boundary)


# ============================================================
# 8. CORRECT AREA DEFINITIONS
# ============================================================

# 1. Area assigned a provisional zoning estimate.
classified_geometry = classified

# 2. Unknown zoning excluding already-classified space.
unknown_only_geometry = unknown.difference(
    classified
)

# 3. No zoning polygon of either type.
covered_geometry = classified.union(unknown)

truly_unmapped_geometry = boundary.difference(
    covered_geometry
)

# Extra diagnostic measurements.
main_unknown_overlap = classified.intersection(
    unknown
)

ambiguous_outside_main = ambiguous.difference(
    classified
)

# Areas in hectares.
kmc_ha = hectares(boundary)

classified_ha = hectares(
    classified_geometry
)

unknown_total_ha = hectares(unknown)

unknown_only_ha = hectares(
    unknown_only_geometry
)

truly_unmapped_ha = hectares(
    truly_unmapped_geometry
)

overlap_ha = hectares(
    main_unknown_overlap
)

ambiguous_ha = hectares(ambiguous)

ambiguous_outside_ha = hectares(
    ambiguous_outside_main
)

old_unmapped_ha = hectares(
    boundary.difference(classified)
)

partition_total = (
    classified_ha
    + unknown_only_ha
    + truly_unmapped_ha
)

partition_difference = abs(
    kmc_ha - partition_total
)


# ============================================================
# 9. SAVE CORRECT COVERAGE CLASSIFICATION
# ============================================================

coverage_records = [
    {
        "category": "CLASSIFIED_ESTIMATES",
        "definition": (
            "Area covered by 508 provisional zoning "
            "estimate polygons"
        ),
        "area_ha": classified_ha,
        "pct_of_kmc": percent(
            classified_ha,
            kmc_ha,
        ),
        "exclusive_partition": True,
    },
    {
        "category": "UNCLASSIFIED_ONLY",
        "definition": (
            "Area covered by 130 unknown zoning "
            "polygons, excluding classified overlap"
        ),
        "area_ha": unknown_only_ha,
        "pct_of_kmc": percent(
            unknown_only_ha,
            kmc_ha,
        ),
        "exclusive_partition": True,
    },
    {
        "category": "TRULY_UNMAPPED",
        "definition": (
            "Area outside both classified and "
            "unclassified zoning polygons"
        ),
        "area_ha": truly_unmapped_ha,
        "pct_of_kmc": percent(
            truly_unmapped_ha,
            kmc_ha,
        ),
        "exclusive_partition": True,
    },
    {
        "category": "AMBIGUOUS_NAMES_OVERLAY",
        "definition": (
            "Additional uncertainty within or "
            "outside the mapped zoning polygons"
        ),
        "area_ha": ambiguous_ha,
        "pct_of_kmc": percent(
            ambiguous_ha,
            kmc_ha,
        ),
        "exclusive_partition": False,
    },
]

coverage = pd.DataFrame(
    coverage_records
)


# ============================================================
# 10. RECONCILE SUPPLIED QA METRICS
# ============================================================

calculated = {
    "kmc_area_ha": kmc_ha,
    "zone_features": len(main_gdf),
    "zone_coverage_ha": classified_ha,
    "zone_coverage_pct": percent(
        classified_ha,
        kmc_ha,
    ),
    "unclassified_features": len(unknown_gdf),
    "unclassified_ha": unknown_only_ha,
    "unclassified_pct": percent(
        unknown_only_ha,
        kmc_ha,
    ),
    "unmapped_ha": truly_unmapped_ha,
    "unmapped_pct": percent(
        truly_unmapped_ha,
        kmc_ha,
    ),
    "ambiguous_class_features": len(
        ambiguous_gdf
    ),
}

count_metrics = {
    "zone_features",
    "unclassified_features",
    "ambiguous_class_features",
}

percentage_metrics = {
    "zone_coverage_pct",
    "unclassified_pct",
    "unmapped_pct",
}

crosscheck_records = []

for metric, value in calculated.items():

    source_value = supplied.get(metric)

    comparable = (
        isinstance(source_value, (int, float))
        and not isinstance(source_value, bool)
        and np.isfinite(source_value)
    )

    if comparable:

        if metric in count_metrics:
            tolerance = 0.0

        elif metric in percentage_metrics:
            tolerance = PERCENT_TOLERANCE

        else:
            tolerance = AREA_TOLERANCE_HA

        difference = abs(
            float(value) - float(source_value)
        )

        matched = difference <= tolerance

    else:

        tolerance = None
        difference = None
        matched = False

    crosscheck_records.append({
        "metric": metric,
        "supplied_value": source_value,
        "recalculated_value": float(value),
        "absolute_difference": difference,
        "tolerance": tolerance,
        "matched": matched,
    })

crosscheck = pd.DataFrame(
    crosscheck_records
)


# ============================================================
# 11. CHECK THE PREVIOUS DISCREPANCY
# ============================================================

previous_unmapped = previous_i.get(
    "unmapped_area_ha"
)

previous_unmapped_pct = previous_i.get(
    "unmapped_pct"
)

correct_old_definition = (
    old_unmapped_ha
    ==
    unknown_only_ha + truly_unmapped_ha
)

# Avoid exact floating-point equality for area values.
correct_old_definition = close_to(
    old_unmapped_ha,
    unknown_only_ha + truly_unmapped_ha,
    AREA_TOLERANCE_HA,
)

previous_value_matches = (
    previous_unmapped is not None
    and close_to(
        old_unmapped_ha,
        previous_unmapped,
        AREA_TOLERANCE_HA,
    )
)

previous_pct_matches = (
    previous_unmapped_pct is not None
    and close_to(
        percent(old_unmapped_ha, kmc_ha),
        previous_unmapped_pct,
        PERCENT_TOLERANCE,
    )
)


# ============================================================
# 12. FINAL VALIDATION
# ============================================================

check(
    "Three exclusive areas cover KMC",
    partition_difference <= AREA_TOLERANCE_HA,
    f"Area difference: {partition_difference:.8f} ha",
)

check(
    "All exclusive category areas nonnegative",
    min(
        classified_ha,
        unknown_only_ha,
        truly_unmapped_ha,
    ) >= 0,
    "No negative areas",
)

check(
    "Main and unknown polygons have no material overlap",
    overlap_ha <= AREA_TOLERANCE_HA,
    f"Overlap: {overlap_ha:.6f} ha",
)

check(
    "Ambiguous layer lies within mapped zoning",
    ambiguous_outside_ha <= AREA_TOLERANCE_HA,
    f"Outside: {ambiguous_outside_ha:.6f} ha",
)

check(
    "All supplied QA metrics reconciled",
    bool(crosscheck["matched"].all()),
    (
        f"{int(crosscheck['matched'].sum())}"
        f"/{len(crosscheck)}"
    ),
)

check(
    "Previous broad unmapped definition reproduced",
    previous_value_matches
    and previous_pct_matches,
    f"Previous: {previous_unmapped} ha",
)

check(
    "Previous difference explained by unknown zoning",
    correct_old_definition,
    (
        f"{old_unmapped_ha:.4f} = "
        f"{unknown_only_ha:.4f} + "
        f"{truly_unmapped_ha:.4f} ha"
    ),
)

check(
    "Frozen Step 16B dataset unchanged",
    sha256(RAW_INPUT) == raw_hash,
    "Source SHA256 unchanged",
)

check(
    "GIS files unchanged after analysis",
    all(
        sha256(path) == source_hashes[path.name]
        for path in source_paths
    ),
    "Read-only GIS analysis",
)


# ============================================================
# 13. SAVE REPORTS
# ============================================================

qa = pd.DataFrame(checks)

status = (
    "PASS"
    if qa["passed"].all()
    else "REVIEW_REQUIRED"
)

coverage.to_csv(
    COVERAGE_CSV,
    index=False,
)

crosscheck.to_csv(
    CROSSCHECK_CSV,
    index=False,
)

qa.to_csv(
    QA_CSV,
    index=False,
)

summary = {
    "step": "16C-J",
    "name": "Zoning Coverage Reconciliation",
    "qa_status": status,
    "generated_utc": datetime.now(
        timezone.utc
    ).isoformat(),
    "reference_crs": f"EPSG:{EPSG}",
    "frozen_input_sha256": raw_hash,
    "gis_source_sha256": source_hashes,
    "kmc_area_ha": kmc_ha,
    "classified_area_ha": classified_ha,
    "unclassified_only_area_ha": unknown_only_ha,
    "truly_unmapped_area_ha": truly_unmapped_ha,
    "previous_broad_unmapped_area_ha":
        old_unmapped_ha,
    "main_unknown_overlap_ha": overlap_ha,
    "ambiguous_name_area_ha": ambiguous_ha,
    "ambiguous_outside_main_ha":
        ambiguous_outside_ha,
    "partition_area_difference_ha":
        partition_difference,
    "reconciled_metric_count": int(
        crosscheck["matched"].sum()
    ),
    "total_compared_metrics": len(crosscheck),
    "qa_checks": len(qa),
    "qa_passed": int(qa["passed"].sum()),
    "interpretation": {
        "previous_issue": (
            "The previous audit labeled all area "
            "outside classified zoning as unmapped."
        ),
        "corrected_definition": (
            "Truly unmapped means outside both "
            "classified and unknown zoning polygons."
        ),
        "legal_status": (
            "All source zoning classifications "
            "remain provisional and unverified."
        ),
        "ambiguous_names": (
            "Ambiguous-name areas are overlapping "
            "uncertainty information, not a fourth "
            "exclusive coverage category."
        ),
        "research_status": (
            "Coverage reconciliation only. "
            "No suitability score, AHP weighting "
            "or grid overlay was produced."
        ),
    },
    "outputs": {
        "coverage": COVERAGE_CSV.name,
        "crosscheck": CROSSCHECK_CSV.name,
        "qa": QA_CSV.name,
        "summary": SUMMARY_JSON.name,
    },
}

with SUMMARY_JSON.open(
    "w",
    encoding="utf-8",
) as file:
    json.dump(
        summary,
        file,
        indent=2,
        ensure_ascii=False,
    )


# ============================================================
# 14. TERMINAL REPORT
# ============================================================

print("\n=== RECONCILED ZONING COVERAGE ===")

print(
    coverage[
        [
            "category",
            "area_ha",
            "pct_of_kmc",
            "exclusive_partition",
        ]
    ].round(4).to_string(index=False)
)

print("\n=== PREVIOUS DISCREPANCY ===")

print(
    "Previously reported unmapped area:",
    round(old_unmapped_ha, 4),
    "ha",
)

print(
    "Unclassified-only area:",
    round(unknown_only_ha, 4),
    "ha",
)

print(
    "Correct truly unmapped area:",
    round(truly_unmapped_ha, 4),
    "ha",
)

print("\n=== SUPPLIED QA RECONCILIATION ===")

print(
    crosscheck[
        [
            "metric",
            "supplied_value",
            "recalculated_value",
            "matched",
        ]
    ].round(4).to_string(index=False)
)

print("\n=== FINAL QA ===")

for _, row in qa.iterrows():

    result = (
        "PASS" if row["passed"]
        else "REVIEW"
    )

    print(
        f"{result}: {row['check']}"
    )

print("\n=== GENERATED REPORTS ===")

for path in [
    COVERAGE_CSV,
    CROSSCHECK_CSV,
    QA_CSV,
    SUMMARY_JSON,
]:
    print(path.relative_to(ROOT))

print("\nOverall QA:", status)

print(
    "Planning data status:",
    "PROVISIONAL / LEGALLY UNVERIFIED"
)

print(
    "No suitability scoring or AHP weighting performed."
)

print("=" * 95)
