from pathlib import Path
import json

import geopandas as gpd
import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

REPORT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "reports"
    / "part2_candidate_wetland"
)

MAP_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "maps"
    / "part2_candidate_wetland"
)


# ============================================================
# INPUTS
# ============================================================

FINAL_TABLE = (
    REPORT_DIR
    / "18_final_candidate_priority_table.csv"
)

FULL_UTM = (
    MAP_DIR
    / "20_final_candidate_priority_utm_repaired.geojson"
)

FULL_WGS84 = (
    MAP_DIR
    / "20_final_candidate_priority_wgs84_repaired.geojson"
)

PRIMARY_UTM = (
    MAP_DIR
    / "20_primary_candidate_layer_utm_repaired.geojson"
)

PRIMARY_WGS84 = (
    MAP_DIR
    / "20_primary_candidate_layer_wgs84_repaired.geojson"
)

REVIEW_UTM = (
    MAP_DIR
    / "20_review_candidate_layer_utm_repaired.geojson"
)

FALSE_POSITIVE_UTM = (
    MAP_DIR
    / "20_manual_false_positive_flags_utm_repaired.geojson"
)


# ============================================================
# OUTPUTS
# ============================================================

FINAL_SUMMARY = (
    REPORT_DIR
    / "21_part2_final_candidate_summary.csv"
)

VALIDATION_AUDIT = (
    REPORT_DIR
    / "21_part2_final_output_validation.json"
)


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n========================================")
    print("PART 2 FINAL OUTPUT VALIDATION")
    print("========================================")

    table = pd.read_csv(FINAL_TABLE)

    full_utm = gpd.read_file(FULL_UTM)
    full_wgs84 = gpd.read_file(FULL_WGS84)

    primary_utm = gpd.read_file(PRIMARY_UTM)
    primary_wgs84 = gpd.read_file(PRIMARY_WGS84)

    review_utm = gpd.read_file(REVIEW_UTM)

    false_positive_utm = gpd.read_file(
        FALSE_POSITIVE_UTM
    )

    checks = {}

    # ========================================================
    # 1. EXPECTED COUNTS
    # ========================================================

    checks["full_table_1097"] = (
        len(table) == 1097
    )

    checks["full_utm_1097"] = (
        len(full_utm) == 1097
    )

    checks["full_wgs84_1097"] = (
        len(full_wgs84) == 1097
    )

    checks["primary_utm_48"] = (
        len(primary_utm) == 48
    )

    checks["primary_wgs84_48"] = (
        len(primary_wgs84) == 48
    )

    checks["review_unreviewed_1005"] = (
        len(review_utm) == 1005
    )

    checks["manual_false_positive_44"] = (
        len(false_positive_utm) == 44
    )

    # ========================================================
    # 2. UNIQUE IDS
    # ========================================================

    datasets = {
        "table": table,
        "full_utm": full_utm,
        "full_wgs84": full_wgs84,
        "primary_utm": primary_utm,
        "primary_wgs84": primary_wgs84,
        "review_utm": review_utm,
        "false_positive_utm": false_positive_utm,
    }

    for name, dataframe in datasets.items():

        checks[
            f"{name}_unique_candidate_ids"
        ] = (
            not dataframe[
                "candidate_id"
            ].duplicated().any()
        )

    # ========================================================
    # 3. FULL UTM / WGS84 ID CONSISTENCY
    # ========================================================

    full_table_ids = set(
        table["candidate_id"]
    )

    full_utm_ids = set(
        full_utm["candidate_id"]
    )

    full_wgs84_ids = set(
        full_wgs84["candidate_id"]
    )

    checks[
        "full_table_utm_id_match"
    ] = (
        full_table_ids
        ==
        full_utm_ids
    )

    checks[
        "full_utm_wgs84_id_match"
    ] = (
        full_utm_ids
        ==
        full_wgs84_ids
    )

    # ========================================================
    # 4. PRIMARY LAYER CONSISTENCY
    # ========================================================

    expected_primary_ids = set(
        table.loc[
            table[
                "include_primary_candidate_layer"
            ].astype(str).str.lower()
            ==
            "true",
            "candidate_id",
        ]
    )

    primary_utm_ids = set(
        primary_utm[
            "candidate_id"
        ]
    )

    primary_wgs84_ids = set(
        primary_wgs84[
            "candidate_id"
        ]
    )

    checks[
        "primary_table_utm_id_match"
    ] = (
        expected_primary_ids
        ==
        primary_utm_ids
    )

    checks[
        "primary_utm_wgs84_id_match"
    ] = (
        primary_utm_ids
        ==
        primary_wgs84_ids
    )

    # ========================================================
    # 5. FALSE-POSITIVE CONSISTENCY
    # ========================================================

    expected_false_positive_ids = set(
        table.loc[
            table[
                "candidate_reporting_status"
            ]
            ==
            "manual_qa_false_positive_flag",
            "candidate_id",
        ]
    )

    false_positive_ids = set(
        false_positive_utm[
            "candidate_id"
        ]
    )

    checks[
        "false_positive_id_match"
    ] = (
        expected_false_positive_ids
        ==
        false_positive_ids
    )

    checks[
        "primary_false_positive_overlap_zero"
    ] = (
        len(
            primary_utm_ids
            &
            false_positive_ids
        )
        ==
        0
    )

    # ========================================================
    # 6. REVIEW LAYER CONSISTENCY
    # ========================================================

    expected_review_ids = set(
        table.loc[
            table[
                "candidate_reporting_status"
            ]
            ==
            "review_priority_unreviewed_candidate",
            "candidate_id",
        ]
    )

    review_ids = set(
        review_utm[
            "candidate_id"
        ]
    )

    checks[
        "review_id_match"
    ] = (
        expected_review_ids
        ==
        review_ids
    )

    # ========================================================
    # 7. GEOMETRY QA
    # ========================================================

    checks[
        "full_utm_all_valid_geometry"
    ] = bool(
        full_utm.geometry.is_valid.all()
    )

    checks[
        "primary_utm_all_valid_geometry"
    ] = bool(
        primary_utm.geometry.is_valid.all()
    )

    checks[
        "review_utm_all_valid_geometry"
    ] = bool(
        review_utm.geometry.is_valid.all()
    )

    checks[
        "false_positive_all_valid_geometry"
    ] = bool(
        false_positive_utm.geometry.is_valid.all()
    )

    checks[
        "full_utm_no_empty_geometry"
    ] = bool(
        (~full_utm.geometry.is_empty).all()
    )

    checks[
        "primary_utm_no_empty_geometry"
    ] = bool(
        (~primary_utm.geometry.is_empty).all()
    )

    # ========================================================
    # 8. CRS QA
    # ========================================================

    checks[
        "full_utm_crs_epsg32644"
    ] = (
        full_utm.crs.to_epsg()
        ==
        32644
    )

    checks[
        "primary_utm_crs_epsg32644"
    ] = (
        primary_utm.crs.to_epsg()
        ==
        32644
    )

    checks[
        "full_wgs84_crs_epsg4326"
    ] = (
        full_wgs84.crs.to_epsg()
        ==
        4326
    )

    checks[
        "primary_wgs84_crs_epsg4326"
    ] = (
        primary_wgs84.crs.to_epsg()
        ==
        4326
    )

    # ========================================================
    # 9. REPORTING-STATUS PARTITION
    # ========================================================

    status_counts = (
        table[
            "candidate_reporting_status"
        ]
        .value_counts()
    )

    status_total = int(
        status_counts.sum()
    )

    checks[
        "reporting_status_partition_1097"
    ] = (
        status_total
        ==
        1097
    )

    # ========================================================
    # 10. PRIMARY SUMMARY
    # ========================================================

    primary_table = (
        table[
            table[
                "candidate_id"
            ]
            .isin(
                primary_utm_ids
            )
        ]
        .copy()
    )

    summary_rows = []

    for priority in [
        "High",
        "Medium",
        "Review",
    ]:

        subset = (
            primary_table[
                primary_table[
                    "screening_priority"
                ]
                ==
                priority
            ]
        )

        summary_rows.append(
            {
                "screening_priority":
                    priority,

                "primary_candidates":
                    len(subset),

                "total_area_ha":
                    float(
                        subset[
                            "area_ha"
                        ].sum()
                    ),

                "mean_probability":
                    (
                        float(
                            subset[
                                "mean_probability"
                            ].mean()
                        )
                        if len(subset) > 0
                        else None
                    ),

                "manually_reviewed":
                    int(
                        subset[
                            "manual_qa_reviewed"
                        ]
                        .astype(str)
                        .str.lower()
                        .eq("true")
                        .sum()
                    ),
            }
        )

    summary = pd.DataFrame(
        summary_rows
    )

    summary.to_csv(
        FINAL_SUMMARY,
        index=False,
    )

    # ========================================================
    # 11. OVERALL PASS / FAIL
    # ========================================================

    failed_checks = [
        name
        for name, result
        in checks.items()
        if not result
    ]

    all_pass = (
        len(failed_checks)
        ==
        0
    )

    # ========================================================
    # 12. AUDIT
    # ========================================================

    audit = {
        "validation_status":
            (
                "PASS"
                if all_pass
                else "FAIL"
            ),

        "checks":
            checks,

        "failed_checks":
            failed_checks,

        "full_candidate_count":
            len(full_utm),

        "primary_candidate_count":
            len(primary_utm),

        "review_unreviewed_count":
            len(review_utm),

        "manual_false_positive_count":
            len(false_positive_utm),

        "important_interpretation":
            (
                "Primary candidates are research screening "
                "outputs, not officially confirmed wetlands."
            ),

        "model_threshold_changed":
            False,

        "model_retrained":
            False,
    }

    with open(
        VALIDATION_AUDIT,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            audit,
            file,
            indent=4,
        )

    # ========================================================
    # TERMINAL OUTPUT
    # ========================================================

    print(
        "\nValidation checks:"
    )

    for name, result in checks.items():

        status = (
            "PASS"
            if result
            else "FAIL"
        )

        print(
            f"{name}: {status}"
        )

    print(
        "\n========================================"
    )

    print(
        "PRIMARY CANDIDATE SUMMARY"
    )

    print(
        "========================================"
    )

    print(
        summary.to_string(
            index=False
        )
    )

    print(
        "\nReporting-status counts:"
    )

    print(
        status_counts.to_string()
    )

    print(
        "\n========================================"
    )

    if all_pass:

        print(
            "FINAL PART 2 OUTPUT VALIDATION: PASS"
        )

    else:

        print(
            "FINAL PART 2 OUTPUT VALIDATION: FAIL"
        )

        print(
            "\nFailed checks:"
        )

        for check in failed_checks:
            print(
                f"- {check}"
            )

    print(
        "========================================"
    )

    print(
        "\nOutputs:"
    )

    print(
        FINAL_SUMMARY
    )

    print(
        VALIDATION_AUDIT
    )

    print(
        "\nIMPORTANT:"
        "\nNo model or candidate-priority decision "
        "was changed during validation."
    )


if __name__ == "__main__":
    main()