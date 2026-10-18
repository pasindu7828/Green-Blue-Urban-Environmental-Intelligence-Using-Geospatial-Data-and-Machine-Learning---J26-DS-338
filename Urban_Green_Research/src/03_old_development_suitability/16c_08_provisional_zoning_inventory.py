
"""
STEP 16C-H - PROVISIONAL UDA GIS INVENTORY

Research Component:
Urban Green Transition, Urbanization Pressure
and Conservation Intelligence

Purpose:
1. Inspect the supplied UDA zoning GIS package.
2. Identify GeoPackage layers and attributes.
3. Check coordinate systems and geometry quality.
4. Record file provenance using SHA256.
5. Save permanent inventory and QA reports.

IMPORTANT:
- GIS datasets are provisional / approximate.
- No spatial overlay is performed.
- No final suitability score is calculated.
- No AHP weighting is performed.
- No changes to frozen research inputs.
"""

from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import sqlite3
import zipfile

import geopandas as gpd
import pandas as pd


# ============================================================
# 1. PROJECT PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = (
    ROOT
    / "data/raw/uda/zoning_provisional"
    / "provided_package_extracted/New data"
)

REPORTS = ROOT / "outputs/reports"

RAW_CSV = (
    ROOT
    / "data/processed/development_suitability"
    / "raw_criteria"
    / "16b_raw_suitability_criteria_100m.csv"
)

PREVIOUS = (
    REPORTS
    / "16c_g_scoring_sensitivity_summary.json"
)


# ============================================================
# 2. EXPECTED GIS SOURCES
# ============================================================

EXPECTED = {
    "Kaduwela_UDA_2024_AUDITED_PROVISIONAL.gpkg":
        "PROVISIONAL_ZONING",

    "Kaduwela_Conservation_Review_Screening_PROVISIONAL.gpkg":
        "PROVISIONAL_CONSERVATION_SCREEN",

    "Kaduwela_2_High_Land_Conservation_Zones_APPROXIMATE.geojson":
        "APPROXIMATE_HIGHLAND_SCREEN",

    "Kaduwela_MC_Boundary.gpkg":
        "SUPPLIED_BOUNDARY_REFERENCE",

    "QA_METRICS.json":
        "SUPPLIED_QA_NOT_INDEPENDENTLY_VERIFIED",
}


# ============================================================
# 3. OUTPUT FILES
# ============================================================

FILES_CSV = (
    REPORTS / "16c_h_source_inventory.csv"
)

LAYERS_CSV = (
    REPORTS / "16c_h_spatial_layer_inventory.csv"
)

FIELDS_CSV = (
    REPORTS / "16c_h_attribute_field_inventory.csv"
)

QA_CSV = (
    REPORTS / "16c_h_zoning_inventory_qa.csv"
)

SUMMARY_JSON = (
    REPORTS / "16c_h_zoning_inventory_summary.json"
)


# ============================================================
# 4. FILE CHECKSUM
# ============================================================

def sha256(path):

    digest = hashlib.sha256()

    with path.open("rb") as source:

        for block in iter(
            lambda: source.read(1024 * 1024),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


# ============================================================
# 5. IDENTIFY GEOPACKAGE LAYERS
# ============================================================

def layers_in_gpkg(path):

    # Read the GeoPackage metadata using SQLite.
    # No changes are written to the source file.

    with sqlite3.connect(str(path)) as conn:

        rows = conn.execute(
            "SELECT table_name "
            "FROM gpkg_contents "
            "WHERE data_type = 'features' "
            "ORDER BY table_name"
        ).fetchall()

    return [row[0] for row in rows]


# ============================================================
# 6. INSPECT GIS LAYER
# ============================================================

def inspect_layer(
    path,
    layer_name,
    source_role,
    field_rows,
):

    if layer_name is None:
        gdf = gpd.read_file(path)

    else:
        gdf = gpd.read_file(
            path,
            layer=layer_name,
        )

    n = len(gdf)

    geom = gdf.geometry

    null = geom.isna()

    empty = geom.is_empty.fillna(False)

    valid = geom.is_valid.fillna(False)

    invalid = (
        ~valid
        & ~null
        & ~empty
    )

    geometry_counts = (
        geom.geom_type
        .fillna("NULL")
        .value_counts()
        .to_dict()
    )

    crs = (
        gdf.crs.to_string()
        if gdf.crs
        else "MISSING"
    )

    epsg = (
        gdf.crs.to_epsg()
        if gdf.crs
        else None
    )

    if n > 0 and (~null).any():

        bounds = [
            float(value)
            for value in gdf.total_bounds
        ]

    else:
        bounds = []

    # Inspect attribute columns.

    for column in gdf.columns:

        if column == geom.name:
            continue

        values = gdf[column]

        examples = (
            values.dropna()
            .astype(str)
            .drop_duplicates()
            .head(3)
            .tolist()
        )

        field_rows.append({
            "file": path.name,
            "layer": (
                layer_name
                if layer_name
                else "(GeoJSON)"
            ),
            "field": column,
            "dtype": str(values.dtype),
            "non_null_count": int(
                values.notna().sum()
            ),
            "unique_values": int(
                values.dropna()
                .astype(str)
                .nunique()
            ),
            "sample_values": " | ".join(
                value[:70]
                for value in examples
            ),
        })

    needs_review = bool(
        n == 0
        or crs == "MISSING"
        or null.any()
        or empty.any()
        or invalid.any()
    )

    return {
        "file": path.name,
        "source_role": source_role,
        "layer": (
            layer_name
            if layer_name
            else "(GeoJSON)"
        ),
        "features": n,
        "crs": crs,
        "epsg": epsg,
        "geometry_types": json.dumps(
            geometry_counts,
            ensure_ascii=False,
        ),
        "null_geometries": int(
            null.sum()
        ),
        "empty_geometries": int(
            empty.sum()
        ),
        "invalid_geometries": int(
            invalid.sum()
        ),
        "attribute_fields": (
            len(gdf.columns) - 1
        ),
        "bounds_native_crs": json.dumps(
            bounds
        ),
        "needs_geometry_review": needs_review,
    }


# ============================================================
# 7. MAIN AUDIT
# ============================================================

def main():

    print("=" * 90)
    print(
        "STEP 16C-H - PROVISIONAL UDA GIS INVENTORY"
    )
    print("=" * 90)

    REPORTS.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # 7.1 VERIFY RESEARCH INPUTS
    # --------------------------------------------------------

    for path in [
        DATA_DIR,
        RAW_CSV,
        PREVIOUS,
    ]:

        if not path.exists():

            raise FileNotFoundError(
                f"Required path missing: {path}"
            )

    with PREVIOUS.open(
        encoding="utf-8"
    ) as file:

        previous = json.load(file)

    checks = []

    def check(name, passed, details):

        checks.append({
            "check": name,
            "passed": bool(passed),
            "details": str(details),
        })

    frozen_sha = sha256(RAW_CSV)

    check(
        "Previous Step 16C-G passed",
        previous.get("qa_status") == "PASS",
        previous.get("qa_status"),
    )

    check(
        "Previous method remains experimental",
        previous.get("methodology_status")
        == "EXPERIMENTAL",
        previous.get("methodology_status"),
    )

    check(
        "Frozen 16B source unchanged",
        previous.get("input_sha256")
        == frozen_sha,
        "SHA256 consistency",
    )

    # --------------------------------------------------------
    # 7.2 INVENTORY SOURCE FILES
    # --------------------------------------------------------

    files = []
    layers = []
    fields = []
    errors = []

    print("\n=== SOURCE FILE INVENTORY ===")

    for filename, role in EXPECTED.items():

        path = DATA_DIR / filename

        exists = path.is_file()

        check(
            "Source exists: " + filename,
            exists,
            "Required file",
        )

        if not exists:
            continue

        print("\nFILE:", filename)

        files.append({
            "file": filename,
            "role": role,
            "relative_path": (
                path.relative_to(ROOT).as_posix()
            ),
            "size_mb": round(
                path.stat().st_size / 1024**2,
                3,
            ),
            "sha256": sha256(path),
        })

        # ----------------------------------------------------
        # 7.3 INSPECT SUPPLIED QA JSON
        # ----------------------------------------------------

        if path.suffix.lower() == ".json":

            try:

                supplied = json.loads(
                    path.read_text(
                        encoding="utf-8-sig"
                    )
                )

                keys = (
                    list(supplied)
                    if isinstance(supplied, dict)
                    else []
                )

                print(
                    "QA JSON top-level keys:",
                    len(keys),
                )

                print(
                    "Sample keys:",
                    keys[:12],
                )

            except Exception as exc:

                errors.append(
                    f"{filename}: invalid JSON: {exc}"
                )

            continue

        # ----------------------------------------------------
        # 7.4 INSPECT SPATIAL LAYERS
        # ----------------------------------------------------

        try:

            if path.suffix.lower() == ".gpkg":

                layer_names = layers_in_gpkg(
                    path
                )

            else:

                layer_names = [None]

            if not layer_names:

                errors.append(
                    f"{filename}: no spatial feature layers"
                )

                continue

            for layer_name in layer_names:

                info = inspect_layer(
                    path,
                    layer_name,
                    role,
                    fields,
                )

                layers.append(info)

                print(
                    "Layer:",
                    info["layer"],
                )

                print(
                    "Features:",
                    f"{info['features']:,}",
                )

                print(
                    "CRS:",
                    info["crs"],
                )

                print(
                    "Geometry types:",
                    info["geometry_types"],
                )

                print(
                    "Invalid geometries:",
                    info["invalid_geometries"],
                )

                print(
                    "Null geometries:",
                    info["null_geometries"],
                )

                print(
                    "Empty geometries:",
                    info["empty_geometries"],
                )

                print(
                    "Attribute fields:",
                    info["attribute_fields"],
                )

        except Exception as exc:

            errors.append(
                f"{filename}: "
                f"{type(exc).__name__}: {exc}"
            )

    # --------------------------------------------------------
    # 7.5 INSPECT ZIP ARCHIVES WITHOUT EXTRACTION
    # --------------------------------------------------------

    zip_paths = [
        (
            DATA_DIR.parent.parent
            / "New data.zip"
        ),
        (
            DATA_DIR
            / "Kaduwela_Zoning_QA_Reviewed_Portable_Package.zip"
        ),
    ]

    print("\n=== ZIP ARCHIVE INVENTORY ===")

    for path in zip_paths:

        if not path.is_file():
            continue

        try:

            with zipfile.ZipFile(path) as archive:

                print(
                    path.name,
                    "- entries:",
                    len(archive.namelist()),
                )

        except Exception as exc:

            errors.append(
                f"ZIP {path.name}: "
                f"{type(exc).__name__}: {exc}"
            )

    # --------------------------------------------------------
    # 7.6 QUALITY ASSURANCE
    # --------------------------------------------------------

    check(
        "All five required source files available",
        len(files) == 5,
        len(files),
    )

    check(
        "Spatial layers readable",
        len(layers) >= 4 and not errors,
        (
            f"{len(layers)} layers; "
            f"errors: {errors}"
        ),
    )

    check(
        "Every spatial layer has CRS and features",
        bool(layers)
        and all(
            row["crs"] != "MISSING"
            and row["features"] > 0
            for row in layers
        ),
        "Metadata integrity",
    )

    check(
        "No null, empty or invalid geometries",
        bool(layers)
        and all(
            not row["needs_geometry_review"]
            for row in layers
        ),
        "Geometry integrity",
    )

    check(
        "Frozen raw data unchanged at end",
        sha256(RAW_CSV) == frozen_sha,
        "Read-only pipeline",
    )

    # --------------------------------------------------------
    # 7.7 SAVE REPORTS
    # --------------------------------------------------------

    qa = pd.DataFrame(checks)

    pd.DataFrame(files).to_csv(
        FILES_CSV,
        index=False,
    )

    pd.DataFrame(layers).to_csv(
        LAYERS_CSV,
        index=False,
    )

    pd.DataFrame(fields).to_csv(
        FIELDS_CSV,
        index=False,
    )

    qa.to_csv(
        QA_CSV,
        index=False,
    )

    status = (
        "PASS"
        if qa["passed"].all()
        else "REVIEW_REQUIRED"
    )

    # --------------------------------------------------------
    # 7.8 SAVE SUMMARY JSON
    # --------------------------------------------------------

    summary = {
        "step": "16C-H",
        "qa_status": status,
        "methodology_status": "GIS_INVENTORY_ONLY",

        "generated_utc": datetime.now(
            timezone.utc
        ).isoformat(),

        "frozen_input_sha256": frozen_sha,

        "files_found": len(files),

        "spatial_layers": len(layers),

        "fields_inspected": len(fields),

        "qa_checks": len(checks),

        "qa_passed": int(
            qa["passed"].sum()
        ),

        "layer_warnings": [
            f"{row['file']}:{row['layer']}"
            for row in layers
            if row["needs_geometry_review"]
        ],

        "read_errors": errors,

        "limitations": [
            (
                "GIS filenames indicate provisional "
                "or approximate data sources."
            ),
            (
                "The supplied QA JSON is not "
                "independent legal verification."
            ),
            (
                "Geometry validity does not prove "
                "legal zoning correctness."
            ),
            (
                "No spatial overlay, suitability score, "
                "AHP weighting, or final criterion "
                "selection was performed."
            ),
            (
                "C5 elevation remains unresolved. "
                "Elevation-based conservation rules "
                "are specific to certain zones."
            ),
        ],
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

    # --------------------------------------------------------
    # 7.9 TERMINAL REPORT
    # --------------------------------------------------------

    print("\n=== STEP 16C-H QA ===")

    for row in checks:

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
        FILES_CSV,
        LAYERS_CSV,
        FIELDS_CSV,
        QA_CSV,
        SUMMARY_JSON,
    ]:

        print(
            path.relative_to(ROOT)
        )

    print("\nOverall QA:", status)

    print(
        "No zoning overlay or final suitability "
        "scoring performed."
    )

    print("=" * 90)


# ============================================================
# 8. EXECUTE
# ============================================================

if __name__ == "__main__":
    main()
