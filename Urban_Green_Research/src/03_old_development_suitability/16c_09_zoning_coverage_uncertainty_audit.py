
"""
STEP 16C-I - ZONING COVERAGE AND UNCERTAINTY AUDIT

Research:
Urban Green Transition, Urbanization Pressure
and Conservation Intelligence

Objectives:
- Measure provisional zoning coverage in Kaduwela.
- Examine unmapped, unclassified and ambiguous areas.
- Compare calculated values with supplied QA metadata.
- Review zoning attribute categories.
- Preserve original GIS and research datasets.

Important:
This is an exploratory source-quality audit.

A valid provisional GIS polygon is NOT proof of
legally approved zoning or development permission.

No AHP, standardization, suitability scoring,
or development permission classification is performed.
"""

from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json

import geopandas as gpd
import pandas as pd
import numpy as np


# ==========================================================
# 1. PATHS
# ==========================================================

ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = (
    ROOT
    / "data/raw/uda/zoning_provisional"
    / "provided_package_extracted/New data"
)

REPORTS = ROOT / "outputs/reports"

PREVIOUS = (
    REPORTS
    / "16c_h_zoning_inventory_summary.json"
)

RAW_CRITERIA = (
    ROOT
    / "data/processed/development_suitability"
    / "raw_criteria"
    / "16b_raw_suitability_criteria_100m.csv"
)

QA_SOURCE = DATA_DIR / "QA_METRICS.json"

COVERAGE_CSV = (
    REPORTS
    / "16c_i_zoning_spatial_coverage.csv"
)

CATEGORY_CSV = (
    REPORTS
    / "16c_i_zoning_attribute_categories.csv"
)

CROSSCHECK_CSV = (
    REPORTS
    / "16c_i_supplied_qa_crosscheck.csv"
)

QA_CSV = (
    REPORTS
    / "16c_i_zoning_coverage_qa.csv"
)

SUMMARY_JSON = (
    REPORTS
    / "16c_i_zoning_coverage_summary.json"
)

TARGET_EPSG = 32644


# ==========================================================
# 2. SPATIAL SOURCES
# ==========================================================

SOURCES = {
    "main_zoning": (
        "Kaduwela_UDA_2024_AUDITED_PROVISIONAL.gpkg",
        "UDA_2024_ZONE_ESTIMATES",
        508,
    ),

    "unknown_zoning": (
        "Kaduwela_Conservation_Review_Screening_PROVISIONAL.gpkg",
        "UNKNOWN_ZONING_130",
        130,
    ),

    "ambiguous_names": (
        "Kaduwela_UDA_2024_AUDITED_PROVISIONAL.gpkg",
        "QA_171_Ambiguous_Names",
        171,
    ),

    "conservation_water": (
        "Kaduwela_Conservation_Review_Screening_PROVISIONAL.gpkg",
        "CONSERVATION_WATER_188",
        188,
    ),

    "highland_approximate": (
        "Kaduwela_2_High_Land_Conservation_Zones_APPROXIMATE.geojson",
        None,
        2,
    ),

    "municipal_boundary": (
        "Kaduwela_MC_Boundary.gpkg",
        "Kaduwela_MC_Boundary",
        1,
    ),
}


# ==========================================================
# 3. HELPER FUNCTIONS
# ==========================================================

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
        "r",
        encoding="utf-8-sig",
    ) as file:
        return json.load(file)


def union_geometry(geometries):
    """
    Combine geometries without adding overlapping
    areas more than once.
    """

    if hasattr(geometries, "union_all"):
        return geometries.union_all()

    return geometries.unary_union


def hectares(area_square_metres):
    return float(area_square_metres / 10000.0)


def percentage(part, whole):
    if whole <= 0:
        return np.nan

    return float(100.0 * part / whole)


def round4(value):
    return round(float(value), 4)


# ==========================================================
# 4. START
# ==========================================================

print("=" * 95)
print("STEP 16C-I - ZONING COVERAGE AND UNCERTAINTY AUDIT")
print("=" * 95)

required = [
    DATA_DIR,
    PREVIOUS,
    RAW_CRITERIA,
    QA_SOURCE,
]

for path in required:
    if not path.exists():
        raise FileNotFoundError(path)

REPORTS.mkdir(
    parents=True,
    exist_ok=True,
)

previous = read_json(PREVIOUS)
supplied_qa = read_json(QA_SOURCE)

original_raw_hash = sha256(RAW_CRITERIA)

checks = []


def check(name, passed, details):
    checks.append({
        "check": name,
        "passed": bool(passed),
        "details": str(details),
    })


check(
    "Previous Step 16C-H passed",
    previous.get("qa_status") == "PASS",
    previous.get("qa_status"),
)

check(
    "Frozen research input unchanged",
    original_raw_hash
    == previous.get("frozen_input_sha256"),
    "SHA256 consistency",
)


# ==========================================================
# 5. LOAD SPATIAL DATA
# ==========================================================

loaded = {}

print("\n=== SPATIAL SOURCE CHECKS ===")

for key, (
    filename,
    layer,
    expected_count,
) in SOURCES.items():

    path = DATA_DIR / filename

    if not path.is_file():
        raise FileNotFoundError(path)

    if layer is None:
        gdf = gpd.read_file(path)
    else:
        gdf = gpd.read_file(
            path,
            layer=layer,
        )

    if gdf.crs is None:
        raise RuntimeError(
            f"Missing CRS: {key}"
        )

    valid = bool(
        gdf.geometry.notna().all()
        and (~gdf.geometry.is_empty).all()
        and gdf.geometry.is_valid.all()
    )

    check(
        f"{key} geometries valid",
        valid,
        f"Features: {len(gdf)}",
    )

    check(
        f"{key} expected count",
        len(gdf) == expected_count,
        (
            f"Expected: {expected_count}; "
            f"Actual: {len(gdf)}"
        ),
    )

    loaded[key] = gdf

    print(
        f"{key:24s} "
        f"features={len(gdf):4d} "
        f"crs={gdf.crs}"
    )

if not all(item["passed"] for item in checks):
    pd.DataFrame(checks).to_csv(
        QA_CSV,
        index=False,
    )
    raise RuntimeError(
        "Initial source checks failed."
    )


# ==========================================================
# 6. PROJECT TO METRES
# ==========================================================

projected = {}

for key, gdf in loaded.items():

    projected[key] = gdf.to_crs(
        epsg=TARGET_EPSG
    )

boundary = union_geometry(
    projected["municipal_boundary"].geometry
)

kmc_area_ha = hectares(
    boundary.area
)

if kmc_area_ha <= 0:
    raise RuntimeError(
        "Invalid Kaduwela boundary area."
    )

print("\nKMC reference area (ha):", round4(kmc_area_ha))


# ==========================================================
# 7. CALCULATE SPATIAL COVERAGE
# ==========================================================

coverage_rows = []
clipped_unions = {}

for key, gdf in projected.items():

    original_geom = gdf.geometry

    original_area = hectares(
        original_geom.area.sum()
    )

    # Clip geometries to the supplied KMC boundary.
    # No GIS file is modified.

    clipped = original_geom.intersection(
        boundary
    )

    valid_clipped = clipped[
        clipped.notna()
        & (~clipped.is_empty)
    ]

    clipped_sum_ha = hectares(
        valid_clipped.area.sum()
    )

    if len(valid_clipped) > 0:
        combined = union_geometry(
            valid_clipped
        )
        union_area_ha = hectares(
            combined.area
        )
    else:
        combined = None
        union_area_ha = 0.0

    clipped_unions[key] = combined

    outside_ha = max(
        0.0,
        original_area - clipped_sum_ha
    )

    overlap_excess_ha = max(
        0.0,
        clipped_sum_ha - union_area_ha
    )

    coverage_rows.append({
        "source": key,
        "features": len(gdf),
        "original_area_sum_ha":
            round4(original_area),
        "within_kmc_area_sum_ha":
            round4(clipped_sum_ha),
        "within_kmc_union_area_ha":
            round4(union_area_ha),
        "outside_kmc_area_sum_ha":
            round4(outside_ha),
        "internal_overlap_excess_ha":
            round4(overlap_excess_ha),
        "kmc_coverage_pct":
            round4(
                percentage(
                    union_area_ha,
                    kmc_area_ha
                )
            ),
        "source_status":
            "PROVISIONAL_OR_REFERENCE",
    })


coverage = pd.DataFrame(
    coverage_rows
)


# ==========================================================
# 8. MAIN ZONING COVERAGE GAPS
# ==========================================================

main_union = clipped_unions[
    "main_zoning"
]

if main_union is None:
    raise RuntimeError(
        "Main zoning dataset has no KMC coverage."
    )

main_coverage_ha = hectares(
    main_union.area
)

unmapped_geometry = boundary.difference(
    main_union
)

unmapped_area_ha = hectares(
    unmapped_geometry.area
)

mapped_pct = percentage(
    main_coverage_ha,
    kmc_area_ha
)

unmapped_pct = percentage(
    unmapped_area_ha,
    kmc_area_ha
)


def get_coverage_area(key):

    row = coverage[
        coverage["source"] == key
    ].iloc[0]

    return float(
        row["within_kmc_union_area_ha"]
    )


unknown_area_ha = get_coverage_area(
    "unknown_zoning"
)

ambiguous_area_ha = get_coverage_area(
    "ambiguous_names"
)

conservation_area_ha = get_coverage_area(
    "conservation_water"
)

highland_area_ha = get_coverage_area(
    "highland_approximate"
)

print("\n=== COVERAGE SUMMARY ===")

print(
    "Main mapped area (ha):",
    round4(main_coverage_ha)
)

print(
    "Main mapped percentage:",
    round4(mapped_pct)
)

print(
    "Unmapped area (ha):",
    round4(unmapped_area_ha)
)

print(
    "Unmapped percentage:",
    round4(unmapped_pct)
)

print(
    "Unknown zoning area (ha):",
    round4(unknown_area_ha)
)

print(
    "Ambiguous-name area (ha):",
    round4(ambiguous_area_ha)
)

print(
    "Conservation/water review area (ha):",
    round4(conservation_area_ha)
)

print(
    "Approximate highland area (ha):",
    round4(highland_area_ha)
)

print(
    "\nNOTE: These categories may overlap. "
    "Do not add their areas together."
)


# ==========================================================
# 9. CHECK SUBSET RELATIONSHIPS
# ==========================================================

subset_review = {}

for key in [
    "unknown_zoning",
    "ambiguous_names",
    "conservation_water",
    "highland_approximate",
]:

    geometry = clipped_unions[key]

    if geometry is None:
        outside_main_ha = 0.0
    else:
        outside_main_ha = hectares(
            geometry.difference(
                main_union
            ).area
        )

    subset_review[key] = round4(
        outside_main_ha
    )

print("\n=== AREAS OUTSIDE MAIN ZONING FOOTPRINT ===")

for key, value in subset_review.items():

    print(
        key,
        ":",
        value,
        "ha"
    )


# ==========================================================
# 10. INSPECT MAIN ZONING ATTRIBUTES
# ==========================================================

main_gdf = loaded["main_zoning"]

category_rows = []

candidate_keywords = [
    "zone",
    "class",
    "name",
    "code",
    "status",
    "flag",
    "category",
    "type",
    "confidence",
    "source",
]

for column in main_gdf.columns:

    if column == main_gdf.geometry.name:
        continue

    column_lower = column.lower()

    if not any(
        keyword in column_lower
        for keyword in candidate_keywords
    ):
        continue

    values = (
        main_gdf[column]
        .fillna("<MISSING>")
        .astype(str)
    )

    counts = values.value_counts(
        dropna=False
    )

    for value, count in counts.head(25).items():

        category_rows.append({
            "layer": "UDA_2024_ZONE_ESTIMATES",
            "field": column,
            "value": value,
            "feature_count": int(count),
            "field_unique_values":
                int(len(counts)),
            "display": "TOP_25_VALUES",
        })

    if len(counts) > 25:

        category_rows.append({
            "layer": "UDA_2024_ZONE_ESTIMATES",
            "field": column,
            "value": "<OTHER_VALUES>",
            "feature_count": int(
                counts.iloc[25:].sum()
            ),
            "field_unique_values":
                int(len(counts)),
            "display": "REMAINING_TOTAL",
        })


categories = pd.DataFrame(
    category_rows,
    columns=[
        "layer",
        "field",
        "value",
        "feature_count",
        "field_unique_values",
        "display",
    ],
)

print("\n=== ZONING ATTRIBUTE FIELDS ===")

if not categories.empty:

    for field in categories["field"].unique():

        field_values = categories[
            categories["field"] == field
        ]

        examples = field_values[
            "value"
        ].head(5).tolist()

        print(
            field,
            "=>",
            examples
        )

else:

    print(
        "No matching classification fields found. "
        "Manual attribute review is needed."
    )


# ==========================================================
# 11. COMPARE SUPPLIED QA METRICS
# ==========================================================

# These calculations are independently produced,
# but the package may use different area definitions.
# Differences are flagged for review rather than
# silently treated as evidence of GIS failure.

calculated = {
    "kmc_area_ha": kmc_area_ha,
    "zone_features": len(main_gdf),
    "zone_coverage_ha": main_coverage_ha,
    "zone_coverage_pct": mapped_pct,
    "unclassified_features":
        len(loaded["unknown_zoning"]),
    "unclassified_ha": unknown_area_ha,
    "unclassified_pct": percentage(
        unknown_area_ha,
        kmc_area_ha,
    ),
    "unmapped_ha": unmapped_area_ha,
    "unmapped_pct": unmapped_pct,
    "ambiguous_class_features":
        len(loaded["ambiguous_names"]),
}

count_metrics = {
    "zone_features",
    "unclassified_features",
    "ambiguous_class_features",
}

crosscheck_rows = []

for key, current_value in calculated.items():

    supplied_value = supplied_qa.get(key)

    if (
        isinstance(supplied_value, (int, float))
        and not isinstance(supplied_value, bool)
    ):

        difference = abs(
            float(current_value)
            - float(supplied_value)
        )

        if key in count_metrics:
            tolerance = 0.0
        else:
            # Review tolerance only.
            # This is not a planning threshold.
            tolerance = max(
                0.5,
                abs(float(supplied_value)) * 0.01,
            )

        match = difference <= tolerance

        result = (
            "WITHIN_REVIEW_TOLERANCE"
            if match
            else "DIFFERENCE_REQUIRES_REVIEW"
        )

    else:

        difference = None
        tolerance = None
        result = "NOT_COMPARABLE"

    crosscheck_rows.append({
        "metric": key,
        "supplied_value": supplied_value,
        "calculated_value": round4(
            current_value
        ),
        "absolute_difference": difference,
        "review_tolerance": tolerance,
        "comparison_status": result,
        "note": (
            "Area differences may reflect "
            "different clipping, overlap or "
            "coverage definitions."
        ),
    })


crosscheck = pd.DataFrame(
    crosscheck_rows
)


# ==========================================================
# 12. FINAL QA
# ==========================================================

check(
    "KMC reference boundary has positive area",
    kmc_area_ha > 0,
    f"{round4(kmc_area_ha)} ha",
)

check(
    "Main zoning coverage not above KMC area",
    main_coverage_ha <= kmc_area_ha + 0.01,
    "Union-area coverage check",
)

check(
    "Mapped and unmapped percentages total 100",
    abs(
        mapped_pct + unmapped_pct - 100
    ) < 0.01,
    "Complementary areas",
)

check(
    "Coverage measurements are nonnegative",
    bool(
        coverage[
            [
                "original_area_sum_ha",
                "within_kmc_area_sum_ha",
                "within_kmc_union_area_ha",
                "outside_kmc_area_sum_ha",
                "internal_overlap_excess_ha",
            ]
        ].ge(0).all().all()
    ),
    "No negative areas",
)

check(
    "All measured percentages are valid",
    bool(
        coverage[
            "kmc_coverage_pct"
        ].between(0, 100.01).all()
    ),
    "Expected range: 0 to 100",
)

check(
    "QA metadata is a dictionary",
    isinstance(supplied_qa, dict),
    "Source QA structure",
)

check(
    "Frozen raw criteria unchanged",
    sha256(RAW_CRITERIA)
    == original_raw_hash,
    "Read-only research audit",
)

qa = pd.DataFrame(
    checks
)

status = (
    "PASS"
    if qa["passed"].all()
    else "REVIEW_REQUIRED"
)


# ==========================================================
# 13. SAVE REPORTS
# ==========================================================

coverage.to_csv(
    COVERAGE_CSV,
    index=False,
)

categories.to_csv(
    CATEGORY_CSV,
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

review_differences = crosscheck[
    crosscheck["comparison_status"]
    == "DIFFERENCE_REQUIRES_REVIEW"
]

summary = {
    "step": "16C-I",
    "name": "Zoning Coverage and Uncertainty Audit",

    "qa_status": status,

    "planning_use_status":
        "PROVISIONAL_REVIEW_REQUIRED",

    "generated_utc": datetime.now(
        timezone.utc
    ).isoformat(),

    "target_epsg": TARGET_EPSG,

    "frozen_raw_sha256":
        original_raw_hash,

    "kmc_area_ha": round4(
        kmc_area_ha
    ),

    "main_zoning_coverage_ha":
        round4(main_coverage_ha),

    "main_zoning_coverage_pct":
        round4(mapped_pct),

    "unmapped_area_ha":
        round4(unmapped_area_ha),

    "unmapped_pct":
        round4(unmapped_pct),

    "unknown_zoning_area_ha":
        round4(unknown_area_ha),

    "ambiguous_area_ha":
        round4(ambiguous_area_ha),

    "conservation_review_area_ha":
        round4(conservation_area_ha),

    "approximate_highland_area_ha":
        round4(highland_area_ha),

    "outside_main_zoning_ha":
        subset_review,

    "supplied_qa_metrics":
        supplied_qa,

    "crosscheck_review_count":
        int(len(review_differences)),

    "qa_checks": len(qa),

    "qa_passed": int(
        qa["passed"].sum()
    ),

    "important_limitations": [
        (
            "All zoning layers are provisional "
            "or approximate reference data."
        ),
        (
            "Unknown and ambiguous zones remain "
            "uncertain even when geometry is valid."
        ),
        (
            "Category areas may overlap and "
            "must not be summed directly."
        ),
        (
            "The supplied QA metrics may use "
            "different area definitions."
        ),
        (
            "No legally approved zoning boundary "
            "is established by this calculation."
        ),
        (
            "No zoning overlay with the 9240 "
            "research grid cells was performed."
        ),
        (
            "No AHP weighting, suitability scoring "
            "or final development recommendation "
            "was produced."
        ),
    ],

    "outputs": {
        "coverage": COVERAGE_CSV.name,
        "categories": CATEGORY_CSV.name,
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
        default=str,
    )


# ==========================================================
# 14. PRINT RESULTS
# ==========================================================

print("\n=== SPATIAL COVERAGE ===")

print(
    coverage[
        [
            "source",
            "features",
            "within_kmc_union_area_ha",
            "kmc_coverage_pct",
            "internal_overlap_excess_ha",
        ]
    ].to_string(index=False)
)

print("\n=== SUPPLIED QA CROSSCHECK ===")

print(
    crosscheck[
        [
            "metric",
            "supplied_value",
            "calculated_value",
            "comparison_status",
        ]
    ].to_string(index=False)
)

print("\n=== FINAL QA ===")

for _, row in qa.iterrows():

    result = (
        "PASS"
        if row["passed"]
        else "REVIEW"
    )

    print(
        f"{result}: {row['check']}"
    )

print("\n=== GENERATED REPORTS ===")

for path in [
    COVERAGE_CSV,
    CATEGORY_CSV,
    CROSSCHECK_CSV,
    QA_CSV,
    SUMMARY_JSON,
]:

    print(
        path.relative_to(ROOT)
    )

print("\nOverall QA:", status)

print(
    "Planning use:",
    "PROVISIONAL - REVIEW REQUIRED"
)

print(
    "Supplied QA metric differences:",
    len(review_differences)
)

print(
    "No suitability scoring or AHP "
    "weighting performed."
)

print("=" * 95)
