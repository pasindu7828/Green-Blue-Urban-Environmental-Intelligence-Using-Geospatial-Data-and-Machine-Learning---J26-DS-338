
"""
J26-DS-338 - Urban Green Research

STEP 5E.2 - Global Explainable AI Service

Reads final Step 15A/15B reports.
No model training, prediction or SHAP recalculation.

SHAP output space: RAW_XGBOOST_MARGIN
Importance percentages: normalized within each group.
Explanations describe model behaviour, not causality.
"""

import csv
import math
from collections import Counter, defaultdict
from copy import deepcopy
from functools import lru_cache

from backend.config import URBAN_GREEN_ROOT


REPORTS = URBAN_GREEN_ROOT / "outputs" / "reports"

GROUPS = {
    "all_scored": {
        "count": 4620,
        "features": "15a_global_shap_importance.csv",
        "feature_rank": "global_importance_rank",
        "domains": "15b_global_shap_domain_importance.csv",
        "domain_rank": "global_domain_rank",
    },
    "threshold_flagged": {
        "count": 255,
        "features": "15a_flagged_global_shap_importance.csv",
        "feature_rank": "flagged_importance_rank",
        "domains": "15b_flagged_shap_domain_importance.csv",
        "domain_rank": "flagged_domain_rank",
    },
}

FEATURE_MAP_FILE = "15b_feature_explanation_domain_map.csv"

QA_FILES = {
    "15a": ("15a_shap_qa.csv", "overall_step15a_status"),
    "15b": (
        "15b_explanation_domain_qa.csv",
        "overall_step15b_status",
    ),
}


def _read_csv(filename):
    path = REPORTS / filename

    if not path.is_file():
        raise FileNotFoundError(path)

    with path.open(
        "r", encoding="utf-8-sig", newline=""
    ) as file:
        reader = csv.DictReader(file)

        if not reader.fieldnames:
            raise ValueError(f"Missing header: {filename}")

        rows = list(reader)

    if not rows:
        raise ValueError(f"Empty report: {filename}")

    return rows


def _number(row, column):
    try:
        value = float(row[column])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            f"Invalid numeric column: {column}"
        ) from exc

    if not math.isfinite(value):
        raise ValueError(f"Non-finite value: {column}")

    return value


def _integer(row, column):
    value = _number(row, column)

    if not value.is_integer():
        raise ValueError(f"Invalid integer: {column}")

    return int(value)


def _check_percentage_total(records, key):
    total = sum(item[key] for item in records)

    if not math.isclose(
        total, 100.0, rel_tol=0, abs_tol=0.01
    ):
        raise ValueError(
            f"Importance percentages do not sum to 100: {key}"
        )


def _check_ranks(records):
    ranks = sorted(item["rank"] for item in records)

    if ranks != list(range(1, len(records) + 1)):
        raise ValueError("Duplicate or invalid XAI ranks")


@lru_cache(maxsize=1)
def _load_xai_data():

    feature_map_rows = _read_csv(FEATURE_MAP_FILE)

    if len(feature_map_rows) != 55:
        raise ValueError("Expected 55 feature mappings")

    feature_map = {}

    for row in feature_map_rows:
        name = row["feature_name"]
        domain = row["explanation_domain"]
        readable = row["human_readable_feature"]

        if not name or not domain or not readable:
            raise ValueError("Incomplete feature mapping")

        if name in feature_map:
            raise ValueError(f"Duplicate feature: {name}")

        feature_map[name] = {
            "human_readable_feature": readable,
            "explanation_domain": domain,
        }

    domain_counts = Counter(
        row["explanation_domain"]
        for row in feature_map_rows
    )

    if len(domain_counts) != 5:
        raise ValueError("Expected five explanation domains")

    qa_status = {}

    for stage, (filename, status_column) in QA_FILES.items():
        qa = _read_csv(filename)

        if (
            len(qa) != 1
            or qa[0].get(status_column) != "PASS"
        ):
            raise ValueError(
                f"Scientific XAI QA failed: {stage}"
            )

        qa_status[stage] = "PASS"

    result = {}

    for group, config in GROUPS.items():

        feature_rows = _read_csv(config["features"])
        domain_rows = _read_csv(config["domains"])

        if len(feature_rows) != 55 or len(domain_rows) != 5:
            raise ValueError(
                f"Unexpected XAI record count: {group}"
            )

        features = []
        seen_features = set()

        for row in feature_rows:
            name = row["feature_name"]

            if name not in feature_map or name in seen_features:
                raise ValueError(
                    f"Invalid feature in {group}: {name}"
                )

            seen_features.add(name)

            record = {
                "rank": _integer(
                    row, config["feature_rank"]
                ),
                "feature_name": name,
                "human_readable_feature": (
                    feature_map[name]["human_readable_feature"]
                ),
                "explanation_domain": (
                    feature_map[name]["explanation_domain"]
                ),
                "mean_absolute_shap_raw_margin": _number(
                    row, "mean_absolute_shap_raw_margin"
                ),
                "importance_percent": _number(
                    row, "importance_percent"
                ),
                "mean_signed_shap_raw_margin": _number(
                    row, "mean_signed_shap_raw_margin"
                ),
            }

            if record["mean_absolute_shap_raw_margin"] < 0:
                raise ValueError("Negative absolute importance")

            if not 0 <= record["importance_percent"] <= 100:
                raise ValueError("Invalid feature percentage")

            # Available for all-scored global importance.
            # Do not invent these fields for flagged records.
            for optional in (
                "mean_positive_shap_raw_margin",
                "mean_negative_shap_raw_margin",
            ):
                if optional in row:
                    record[optional] = _number(row, optional)

            features.append(record)

        if seen_features != set(feature_map):
            raise ValueError("Feature mapping coverage mismatch")

        features.sort(key=lambda item: item["rank"])

        _check_ranks(features)
        _check_percentage_total(
            features, "importance_percent"
        )

        domains = []
        seen_domains = set()

        for row in domain_rows:
            name = row["explanation_domain"]

            if name in seen_domains or name not in domain_counts:
                raise ValueError(f"Invalid domain: {name}")

            seen_domains.add(name)

            record = {
                "rank": _integer(row, config["domain_rank"]),
                "explanation_domain": name,
                "feature_count": _integer(
                    row, "feature_count"
                ),
                "mean_absolute_shap_sum": _number(
                    row, "mean_absolute_shap_sum"
                ),
                "mean_signed_shap_sum": _number(
                    row, "mean_signed_shap_sum"
                ),
                "feature_importance_percent_sum": _number(
                    row, "feature_importance_percent_sum"
                ),
                "absolute_domain_importance_percent": _number(
                    row, "absolute_domain_importance_percent"
                ),
            }

            if record["feature_count"] != domain_counts[name]:
                raise ValueError(
                    f"Domain feature count mismatch: {name}"
                )

            if record["mean_absolute_shap_sum"] < 0:
                raise ValueError("Negative domain magnitude")

            domains.append(record)

        if seen_domains != set(domain_counts):
            raise ValueError("Missing explanation domain")

        domains.sort(key=lambda item: item["rank"])

        _check_ranks(domains)
        _check_percentage_total(
            domains, "absolute_domain_importance_percent"
        )

        grouped = defaultdict(float)

        for feature in features:
            grouped[
                feature["explanation_domain"]
            ] += feature["importance_percent"]

        for domain in domains:
            name = domain["explanation_domain"]

            if not math.isclose(
                grouped[name],
                domain["feature_importance_percent_sum"],
                rel_tol=0,
                abs_tol=0.01,
            ):
                raise ValueError(
                    f"Feature/domain percentage mismatch: {name}"
                )

        result[group] = {
            "explained_grid_count": config["count"],
            "features": features,
            "domains": domains,
        }

    return {
        "groups": result,
        "feature_mapping_count": len(feature_map),
        "research_qa": qa_status,
    }


def _get_group(group):
    if group not in GROUPS:
        raise ValueError(
            "group must be 'all_scored' or 'threshold_flagged'"
        )

    return _load_xai_data()["groups"][group]


def _common_metadata(group):
    return {
        "group": group,
        "explained_grid_count": GROUPS[group]["count"],
        "origin_year": 2025,
        "pressure_period": "2026-2028",
        "shap_output_space": "RAW_XGBOOST_MARGIN",
        "importance_basis": "NORMALIZED_MEAN_ABSOLUTE_SHAP",
        "percentages_normalized_within_group": True,
        "shap_is_probability_contribution": False,
        "explanations_are_causal_claims": False,
        "mode": "read-only",
    }


def get_pressure_xai_overview():
    data = _load_xai_data()

    groups = {}

    for group in GROUPS:
        records = data["groups"][group]

        groups[group] = {
            "explained_grid_count": (
                records["explained_grid_count"]
            ),
            "feature_count": len(records["features"]),
            "domain_count": len(records["domains"]),
            "top_feature": (
                records["features"][0]["feature_name"]
            ),
            "top_domain": (
                records["domains"][0]["explanation_domain"]
            ),
        }

    return {
        "status": "ok",
        "origin_year": 2025,
        "pressure_period": "2026-2028",
        "shap_output_space": "RAW_XGBOOST_MARGIN",
        "explanations_are_causal_claims": False,
        "shap_is_probability_contribution": False,
        "groups": groups,
        "research_qa": dict(data["research_qa"]),
    }


def get_pressure_xai_features(
    group="all_scored",
    limit=10,
):
    records = _get_group(group)

    if not isinstance(limit, int) or isinstance(limit, bool):
        raise ValueError("Invalid feature limit")

    if not 1 <= limit <= 55:
        raise ValueError("Feature limit must be 1-55")

    return {
        **_common_metadata(group),
        "total_features": 55,
        "returned_features": limit,
        "features": deepcopy(records["features"][:limit]),
    }


def get_pressure_xai_domains(group="all_scored"):
    records = _get_group(group)

    return {
        **_common_metadata(group),
        "total_domains": 5,
        "domains": deepcopy(records["domains"]),
    }
