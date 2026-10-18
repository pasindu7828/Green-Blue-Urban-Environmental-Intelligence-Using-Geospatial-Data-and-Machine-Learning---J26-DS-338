
from pathlib import Path
import ast

SOURCE = Path(
    "src/03_development_suitability/16b_raw_suitability_criteria.py"
)

BACKUP = Path(
    "outputs/backups/16b_before_terrain_fallback/"
    "16b_raw_suitability_criteria.py"
)

BASELINE_CSV = Path(
    "outputs/backups/16b_before_terrain_fallback/"
    "16b_raw_suitability_criteria_100m.csv"
)

if not SOURCE.is_file() or not BACKUP.is_file():
    raise FileNotFoundError("Original script or backup is missing.")

if not BASELINE_CSV.is_file():
    raise FileNotFoundError("Baseline CSV backup is missing.")

source = SOURCE.read_text(encoding="utf-8-sig")
backup = BACKUP.read_text(encoding="utf-8-sig")

if source != backup:
    raise RuntimeError(
        "Current source differs from the saved backup. "
        "No modifications made."
    )


def replace_once(old, new, description):
    global source
    count = source.count(old)
    if count != 1:
        raise RuntimeError(
            f"{description}: expected one anchor, found {count}. "
            "No modifications made."
        )
    source = source.replace(old, new, 1)


replace_once(
    "from shapely.geometry import mapping\n",
    "from shapely.geometry import Point, mapping\n",
    "Import Point",
)

replace_once(
    "EXPECTED_GRID_COUNT = 9240\n",
    "EXPECTED_GRID_COUNT = 9240\n"
    "EXPECTED_TERRAIN_FALLBACK_COUNT = 39\n"
    "TERRAIN_FALLBACK_MAX_DISTANCE_M = 30.0\n",
    "Terrain fallback settings",
)

helper = '''
def nearest_valid_gedtm_proxy(
    geometry,
    dem,
    kmc_boundary,
    search_radius_m,
):
    """
    Find the nearest GEDTM pixel centre with jointly valid
    elevation and derived slope.

    This is a flagged local proxy, NOT an in-polygon
    zonal measurement.
    """
    region = geometry.buffer(search_radius_m)

    candidate_mask = geometry_mask(
        [mapping(region)],
        out_shape=dem["elevation"].shape,
        transform=dem["transform"],
        invert=True,
        all_touched=True,
    )

    valid = (
        candidate_mask
        & np.isfinite(dem["elevation"])
        & np.isfinite(dem["slope_deg"])
    )

    best = None

    for row, col in np.argwhere(valid):
        row = int(row)
        col = int(col)

        x, y = dem["transform"] * (
            col + 0.5,
            row + 0.5,
        )

        point = Point(float(x), float(y))
        distance = float(geometry.distance(point))

        if distance > search_radius_m:
            continue

        if not kmc_boundary.covers(point):
            continue

        if best is None or (
            distance, row, col
        ) < (
            best["distance_m"],
            best["source_row"],
            best["source_col"],
        ):
            best = {
                "distance_m": distance,
                "source_row": row,
                "source_col": col,
                "source_x_m": float(x),
                "source_y_m": float(y),
                "elevation_m": float(
                    dem["elevation"][row, col]
                ),
                "slope_deg": float(
                    dem["slope_deg"][row, col]
                ),
            }

    return best


'''

replace_once(
    "def summarize_array_by_geometry(\n",
    helper + "def summarize_array_by_geometry(\n",
    "Nearest-valid GEDTM helper",
)

replace_once(
    "terrain_records = []\n",
    "terrain_records = []\n"
    "terrain_fallback_records = []\n",
    "Terrain provenance records",
)

fallback_logic = '''
    # Apply a local proxy ONLY when the original
    # zonal extraction could not measure terrain.
    needs_fallback = (
        elevation_stats["count"] == 0
        or slope_stats["count"] == 0
    )

    proxy = None

    if needs_fallback:
        cell_row = grid_clip.iloc[i - 1]

        if not (
            elevation_stats["count"] == 0
            and slope_stats["count"] == 0
        ):
            raise RuntimeError(
                "Elevation/slope coverage disagreement: "
                + str(cell_row["grid_id"])
            )

        if not bool(cell_row["boundary_cell"]):
            raise RuntimeError(
                "Terrain fallback requested outside boundary: "
                + str(cell_row["grid_id"])
            )

        if str(cell_row["coverage_group"]) != "E_<25%":
            raise RuntimeError(
                "Unexpected boundary coverage group: "
                + str(cell_row["grid_id"])
            )

        proxy = nearest_valid_gedtm_proxy(
            geometry=geom,
            dem=dem,
            kmc_boundary=kmc_union,
            search_radius_m=TERRAIN_FALLBACK_MAX_DISTANCE_M,
        )

        if proxy is None:
            raise RuntimeError(
                "No acceptable GEDTM proxy for "
                + str(cell_row["grid_id"])
            )

        # These summary fields are local proxy estimates.
        # They are NOT observed within-cell distributions.
        for statistic in (
            "mean", "median", "min", "max", "p90"
        ):
            elevation_stats[statistic] = proxy["elevation_m"]
            slope_stats[statistic] = proxy["slope_deg"]

        # Keep original zonal pixel counts equal to zero.
        # No raster pixels were sampled inside this fragment.

    terrain_fallback_records.append({
        "terrain_fallback_applied": bool(needs_fallback),
        "terrain_extraction_method": (
            "NEAREST_VALID_GEDTM_PROXY"
            if needs_fallback
            else "ZONAL_GEDTM"
        ),
        "terrain_fallback_distance_m": (
            proxy["distance_m"] if proxy else np.nan
        ),
        "terrain_fallback_source_row": (
            proxy["source_row"] if proxy else -1
        ),
        "terrain_fallback_source_col": (
            proxy["source_col"] if proxy else -1
        ),
        "terrain_fallback_source_x_m": (
            proxy["source_x_m"] if proxy else np.nan
        ),
        "terrain_fallback_source_y_m": (
            proxy["source_y_m"] if proxy else np.nan
        ),
    })

'''

replace_once(
    "    terrain_records.append(\n",
    fallback_logic + "    terrain_records.append(\n",
    "Fallback application in terrain loop",
)

replace_once(
    "terrain_df = pd.DataFrame(\n    terrain_records\n)\n",
    "terrain_df = pd.DataFrame(\n"
    "    terrain_records\n"
    ")\n"
    "terrain_df = pd.concat(\n"
    "    [terrain_df, pd.DataFrame(terrain_fallback_records)],\n"
    "    axis=1,\n"
    ")\n",
    "Attach terrain provenance",
)

preservation_qa = '''
# Verify the fallback against the untouched Step 16B backup.
baseline_path = (
    PROJECT_ROOT
    / "outputs"
    / "backups"
    / "16b_before_terrain_fallback"
    / "16b_raw_suitability_criteria_100m.csv"
)

baseline = pd.read_csv(baseline_path)

if (
    len(baseline) != EXPECTED_GRID_COUNT
    or not baseline["grid_id"].is_unique
):
    raise RuntimeError("Baseline CSV grid integrity failed.")

old_missing = (
    baseline["c4_slope_mean_deg"].isna()
    | baseline["c5_elevation_mean_m"].isna()
)

expected_fallback_ids = set(
    baseline.loc[old_missing, "grid_id"]
)

actual_fallback_ids = set(
    grid_clip.loc[
        grid_clip["terrain_fallback_applied"], "grid_id"
    ]
)

if (
    len(expected_fallback_ids)
    != EXPECTED_TERRAIN_FALLBACK_COUNT
    or expected_fallback_ids != actual_fallback_ids
):
    raise RuntimeError(
        "Fallback grid IDs differ from original 39 cells."
    )

old = baseline.set_index("grid_id")
new = grid_clip.set_index("grid_id")

if set(old.index) != set(new.index):
    raise RuntimeError("Grid IDs changed from baseline.")

valid_ids = old.index.difference(
    list(expected_fallback_ids)
)

if len(valid_ids) != (
    EXPECTED_GRID_COUNT - EXPECTED_TERRAIN_FALLBACK_COUNT
):
    raise RuntimeError("Original valid-cell count changed.")

original_terrain_columns = [
    c for c in baseline.columns
    if c.startswith(("c4_", "c5_"))
]

for column in original_terrain_columns:
    before = pd.to_numeric(
        old.loc[valid_ids, column],
        errors="coerce",
    ).to_numpy(dtype=float)

    after = pd.to_numeric(
        new.loc[valid_ids, column],
        errors="coerce",
    ).to_numpy(dtype=float)

    if not np.allclose(
        before,
        after,
        rtol=0.0,
        atol=1e-8,
        equal_nan=True,
    ):
        raise RuntimeError(
            "Previously valid terrain values changed: "
            + column
        )

fallback_rows = grid_clip.loc[
    grid_clip["terrain_fallback_applied"]
]

terrain_fallback_count = len(fallback_rows)

terrain_fallback_max_distance = float(
    fallback_rows["terrain_fallback_distance_m"].max()
)

terrain_fallback_qa_ok = bool(
    terrain_fallback_count
    == EXPECTED_TERRAIN_FALLBACK_COUNT
    and fallback_rows["boundary_cell"].all()
    and (
        fallback_rows["terrain_fallback_distance_m"]
        <= TERRAIN_FALLBACK_MAX_DISTANCE_M
    ).all()
    and (
        fallback_rows["c4_slope_pixel_count"] == 0
    ).all()
    and (
        fallback_rows["c5_elevation_pixel_count"] == 0
    ).all()
    and np.isfinite(
        grid_clip["c4_slope_mean_deg"]
    ).all()
    and np.isfinite(
        grid_clip["c5_elevation_mean_m"]
    ).all()
)

if not terrain_fallback_qa_ok:
    raise RuntimeError("Terrain fallback QA failed.")

print(
    "Terrain fallback QA PASS: "
    f"{terrain_fallback_count} boundary proxies; "
    f"{len(valid_ids)} original valid cells preserved."
)

'''

replace_once(
    "overall_pass = all(\n",
    preservation_qa + "overall_pass = all(\n",
    "Baseline preservation QA",
)

replace_once(
    "        slope_nonnegative,\n"
    "        no_forbidden_outputs,\n",
    "        slope_nonnegative,\n"
    "        terrain_fallback_qa_ok,\n"
    "        no_forbidden_outputs,\n",
    "Include fallback QA in PASS decision",
)

replace_once(
    '            "slope_nonnegative": slope_nonnegative,\n',
    '            "slope_nonnegative": slope_nonnegative,\n'
    '            "terrain_fallback_count": terrain_fallback_count,\n'
    '            "terrain_fallback_qa_ok": terrain_fallback_qa_ok,\n'
    '            "original_valid_terrain_preserved": True,\n'
    '            "terrain_fallback_max_distance_m": '
    'terrain_fallback_max_distance,\n',
    "Extend QA report",
)

audit_save = '''
fallback_audit_path = (
    REPORT_DIR / "16b_terrain_fallback_audit.csv"
)

fallback_audit_columns = [
    "grid_id",
    "boundary_cell",
    "coverage_group",
    "kmc_overlap_area_m2",
    "kmc_overlap_fraction",
    "terrain_extraction_method",
    "terrain_fallback_distance_m",
    "terrain_fallback_source_row",
    "terrain_fallback_source_col",
    "terrain_fallback_source_x_m",
    "terrain_fallback_source_y_m",
    "c4_slope_pixel_count",
    "c4_slope_mean_deg",
    "c5_elevation_pixel_count",
    "c5_elevation_mean_m",
]

fallback_rows[fallback_audit_columns].to_csv(
    fallback_audit_path,
    index=False,
)

'''

replace_once(
    "road_class_df.to_csv(\n",
    audit_save + "road_class_df.to_csv(\n",
    "Save fallback audit report",
)

replace_once(
    '    "name": "Raw 100 m Development Suitability Criteria Generation",\n',
    '    "name": "Raw 100 m Development Suitability Criteria Generation",\n'
    '    "terrain_fallback_count": terrain_fallback_count,\n'
    '    "terrain_fallback_method": "NEAREST_VALID_GEDTM_PROXY",\n'
    '    "terrain_fallback_max_allowed_distance_m": '
    'TERRAIN_FALLBACK_MAX_DISTANCE_M,\n'
    '    "terrain_fallback_max_actual_distance_m": '
    'terrain_fallback_max_distance,\n'
    '    "original_valid_terrain_preserved": True,\n'
    '    "terrain_fallback_note": '
    '"Proxy values are not direct in-polygon zonal measurements. '
    'Fallback cells have zero original zonal pixel count; '
    'their single-point range and percentile fields are proxy '
    'summaries, not observed within-cell distributions.",\n',
    "Summary metadata",
)

replace_once(
    '    "outputs": {\n',
    '    "outputs": {\n'
    '        "terrain_fallback_audit": relative_path(\n'
    '            REPORT_DIR / "16b_terrain_fallback_audit.csv"\n'
    '        ),\n',
    "Register fallback audit output",
)

# Validate the complete modified Python source before writing.
ast.parse(source, filename=str(SOURCE))

SOURCE.write_text(
    source,
    encoding="utf-8",
)

print("=" * 70)
print("STEP 16B TERRAIN CORRECTION PATCH")
print("=" * 70)
print("Status                   : PATCH APPLIED")
print("Python syntax            : PASS")
print("Original backup retained : YES")
print("Primary GEDTM changed    : NO")
print("Step 16B pipeline run    : NO")
print("Research outputs changed : NO")
print("Step 16C started         : NO")
print()
print("Updated:", SOURCE)
print("Next: Review patch output before running Step 16B.")
