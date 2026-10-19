
"""
J26-DS-338 - Urban Green Research

Section 02: Urbanization Pressure and XAI

Read-only access to final Step 14B, 14C, 14D,
and Step 15B research outputs.

No model training, new inference, threshold changes,
SHAP recalculation, or research file modifications.
"""

import csv
import math
import re

from collections import Counter
from copy import deepcopy
from functools import lru_cache

from backend.config import URBAN_GREEN_ROOT


PRESSURE_ROOT = (
    URBAN_GREEN_ROOT
    / "data" / "processed"
    / "urbanization_pressure"
)

REPORTS_ROOT = (
    URBAN_GREEN_ROOT / "outputs" / "reports"
)

SOURCES = {
    "eligibility": (
        PRESSURE_ROOT / "inference"
        / "14b_2025_inference_eligibility.csv"
    ),
    "scores": (
        PRESSURE_ROOT / "inference"
        / "14c_2026_2028_urbanization_pressure_scores.csv"
    ),
    "categories": (
        PRESSURE_ROOT / "inference"
        / "14d_2026_2028_pressure_categories.csv"
    ),
    "explanations": (
        PRESSURE_ROOT / "xai"
        / "15b_dashboard_grid_explanations.csv"
    ),
}

QA_REPORTS = {
    "14c": (
        REPORTS_ROOT / "14c_pressure_scoring_qa.csv",
        "overall_step14c_status",
    ),
    "14d": (
        REPORTS_ROOT / "14d_pressure_category_qa.csv",
        "overall_step14d_status",
    ),
    "15a": (
        REPORTS_ROOT / "15a_shap_qa.csv",
        "overall_step15a_status",
    ),
    "15b": (
        REPORTS_ROOT / "15b_explanation_domain_qa.csv",
        "overall_step15b_status",
    ),
}

BOOLEAN_COLUMNS = {
    "spatial_overlap_eligible",
    "coverage_eligible",
    "current_nonbuilt_eligible",
    "past_stability_eligible",
    "initial_inference_eligible",
    "predictor_complete",
    "final_inference_ready",
    "threshold_flag",
}

INTEGER_PATTERN = re.compile(
    r"^[+-]?(?:0|[1-9][0-9]*)$"
)


def _convert(column, value):
    if value is None:
        return None

    value = value.strip()
    lower = value.lower()

    if not value or lower in {"nan", "null", "none"}:
        return None

    if column in BOOLEAN_COLUMNS:
        if lower in {"true", "1", "1.0"}:
            return True
        if lower in {"false", "0", "0.0"}:
            return False
        raise ValueError(
            f"Invalid boolean {column}: {value}"
        )

    if lower in {"true", "false"}:
        return lower == "true"

    if INTEGER_PATTERN.fullmatch(value):
        return int(value)

    try:
        number = float(value)
    except ValueError:
        return value

    return number if math.isfinite(number) else None


def _read_csv(path):
    if not path.is_file():
        raise FileNotFoundError(path)

    with path.open(
        "r", encoding="utf-8-sig", newline=""
    ) as file:
        reader = csv.DictReader(file)

        if not reader.fieldnames:
            raise ValueError(
                f"Missing CSV header: {path.name}"
            )

        return [
            {
                col: _convert(col, value)
                for col, value in row.items()
            }
            for row in reader
        ]


def _index_by_grid(path):
    rows = _read_csv(path)
    index = {}

    for row in rows:
        grid_id = row.get("grid_id")

        if not isinstance(grid_id, str) or not grid_id:
            raise ValueError(
                f"Invalid grid_id in {path.name}"
            )

        if grid_id in index:
            raise ValueError(
                f"Duplicate grid_id: {grid_id}"
            )

        index[grid_id] = row

    return index


def _assert_same_score(a, b, grid_id):
    if not math.isclose(
        a["urbanization_pressure_score"],
        b["urbanization_pressure_score"],
        rel_tol=0,
        abs_tol=1e-8,
    ):
        raise ValueError(
            f"Score mismatch for grid {grid_id}"
        )


@lru_cache(maxsize=1)
def _load_pressure_data():
    eligibility = _index_by_grid(
        SOURCES["eligibility"]
    )
    scores = _index_by_grid(
        SOURCES["scores"]
    )
    categories = _index_by_grid(
        SOURCES["categories"]
    )
    explanations = _index_by_grid(
        SOURCES["explanations"]
    )

    if len(eligibility) != 9240:
        raise ValueError(
            "Expected 9240 permanent grid cells"
        )

    if not all(
        len(table) == 4620
        for table in (
            scores, categories, explanations
        )
    ):
        raise ValueError(
            "Expected 4620 scored and explained cells"
        )

    ready_ids = {
        grid_id
        for grid_id, row in eligibility.items()
        if row["final_inference_ready"] is True
    }

    scored_ids = set(scores)

    if not (
        ready_ids
        == scored_ids
        == set(categories)
        == set(explanations)
    ):
        raise ValueError(
            "Pressure and XAI grid coverage mismatch"
        )

    qa_status = {}

    for stage, (path, status_column) in (
        QA_REPORTS.items()
    ):
        rows = _read_csv(path)

        if (
            len(rows) != 1
            or rows[0].get(status_column) != "PASS"
        ):
            raise ValueError(
                f"Research QA failed: Step {stage}"
            )

        qa_status[stage] = "PASS"

    category_counts = Counter()

    for grid_id, score in scores.items():
        category = categories[grid_id]
        explanation = explanations[grid_id]

        for row in (score, category, explanation):
            if (
                row["origin_year"] != 2025
                or row["pressure_period"] != "2026-2028"
            ):
                raise ValueError(
                    f"Period mismatch: {grid_id}"
                )

        _assert_same_score(
            score, category, grid_id
        )
        _assert_same_score(
            score, explanation, grid_id
        )

        if (
            score["threshold_flag"]
            is not category["threshold_flag"]
            or score["threshold_flag"]
            is not explanation["threshold_flag"]
        ):
            raise ValueError(
                f"Threshold flag mismatch: {grid_id}"
            )

        if (
            category["pressure_category"]
            != explanation["pressure_category"]
        ):
            raise ValueError(
                f"Category mismatch: {grid_id}"
            )

        if not math.isclose(
            score["operating_threshold"],
            0.715,
            rel_tol=0,
            abs_tol=1e-10,
        ):
            raise ValueError(
                f"Locked threshold mismatch: {grid_id}"
            )

        if (
            explanation["shap_output_space"]
            != "RAW_XGBOOST_MARGIN"
            or explanation["explanation_scope"]
            != "MODEL_EXPLANATION_NOT_CAUSAL_CLAIM"
        ):
            raise ValueError(
                f"XAI interpretation mismatch: {grid_id}"
            )

        category_counts[
            category["pressure_category"]
        ] += 1

    flagged = sum(
        row["threshold_flag"] is True
        for row in scores.values()
    )

    if flagged != 255:
        raise ValueError(
            f"Expected 255 threshold flags, found {flagged}"
        )

    expected_categories = {
        "LOW_RELATIVE": 2310,
        "MODERATE_RELATIVE": 1155,
        "ELEVATED_RELATIVE": 693,
        "HIGH_RELATIVE": 207,
        "THRESHOLD_FLAGGED": 255,
    }

    if dict(category_counts) != expected_categories:
        raise ValueError(
            "Pressure category counts mismatch"
        )

    return {
        "eligibility": eligibility,
        "scores": scores,
        "categories": categories,
        "explanations": explanations,
        "category_counts": expected_categories,
        "qa_status": qa_status,
    }


def get_pressure_overview():
    data = _load_pressure_data()

    return {
        "status": "ok",
        "total_grid_cells": 9240,
        "eligible_scored_cells": 4620,
        "outside_inference_domain_cells": 4620,
        "threshold_flagged_cells": 255,
        "origin_year": 2025,
        "pressure_period": "2026-2028",
        "prediction_window_type": "SINGLE_THREE_YEAR_WINDOW",
        "production_model": "XGBoost",
        "locked_operating_threshold": 0.715,
        "score_is_calibrated_probability": False,
        "shap_output_space": "RAW_XGBOOST_MARGIN",
        "shap_is_causal_explanation": False,
        "category_counts": dict(
            data["category_counts"]
        ),
        "research_qa": dict(data["qa_status"]),
        "mode": "read-only",
    }


def get_pressure_grid(grid_id):
    """
    Return the original research records for one grid.

    Ineligible grids have no model score,
    category record or SHAP explanation.
    """
    data = _load_pressure_data()

    eligibility = data["eligibility"].get(grid_id)

    if eligibility is None:
        raise KeyError(
            f"Unknown pressure grid: {grid_id}"
        )

    ready = eligibility["final_inference_ready"]

    return {
        "grid_id": grid_id,
        "origin_year": 2025,
        "pressure_period": "2026-2028",
        "inference_ready": ready,
        "display_status": (
            "IN_INFERENCE_DOMAIN"
            if ready
            else "NOT_IN_INFERENCE_DOMAIN"
        ),
        "eligibility": deepcopy(eligibility),
        "score": deepcopy(
            data["scores"].get(grid_id)
        ),
        "category": deepcopy(
            data["categories"].get(grid_id)
        ),
        "explanation": deepcopy(
            data["explanations"].get(grid_id)
        ),
    }


def get_pressure_grids(
    offset=0,
    limit=50,
    inference_ready=None,
):
    """
    Paginated records for future dashboard use.

    Filtering never assigns scores to ineligible cells.
    """
    if offset < 0 or not 1 <= limit <= 200:
        raise ValueError(
            "Invalid pressure pagination"
        )

    data = _load_pressure_data()

    ids = sorted(data["eligibility"])

    if inference_ready is not None:
        ids = [
            grid_id
            for grid_id in ids
            if (
                data["eligibility"][grid_id][
                    "final_inference_ready"
                ]
                is inference_ready
            )
        ]

    total = len(ids)

    selected = ids[offset:offset + limit]

    return {
        "total": total,
        "offset": offset,
        "limit": limit,
        "data": [
            get_pressure_grid(grid_id)
            for grid_id in selected
        ],
    }
