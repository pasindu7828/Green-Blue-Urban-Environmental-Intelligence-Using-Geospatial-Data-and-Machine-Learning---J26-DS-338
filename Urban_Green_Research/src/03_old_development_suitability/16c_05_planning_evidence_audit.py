
"""
STEP 16C-E - OFFICIAL PLANNING EVIDENCE AUDIT

Research Component:
Urban Green Transition, Urbanization Pressure and
Conservation Intelligence

Purpose:
1. Review the Kaduwela Development Plan 2023-2033.
2. Record planning evidence related to C1-C5.
3. Save source page numbers and research implications.
4. Validate document references.
5. Generate permanent CSV and JSON reports.

Important:
- No final criterion selection.
- No normalization or standardization.
- No AHP weighting.
- No suitability scoring.
- No zoning overlay.
- Frozen research inputs are not modified.

The PDF is in Sinhala.
Automated text extraction may be imperfect.
Legal details must be checked against the original PDF.
"""

from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json

import pandas as pd

try:
    import fitz

except ImportError as exc:
    raise SystemExit(
        "PyMuPDF is required. Install using: "
        "python -m pip install pymupdf"
    ) from exc


# ============================================================
# 1. PROJECT PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

PDF = (
    ROOT
    / "data"
    / "raw"
    / "uda"
    / "development_plan"
    / "KaduwelaDP2023-2033_S.pdf"
)

RAW_CSV = (
    ROOT
    / "data"
    / "processed"
    / "development_suitability"
    / "raw_criteria"
    / "16b_raw_suitability_criteria_100m.csv"
)

REPORTS = ROOT / "outputs" / "reports"

PREVIOUS = (
    REPORTS
    / "16c_d_criteria_selection_summary.json"
)

REGISTRY_CSV = (
    REPORTS
    / "16c_e_planning_evidence_registry.csv"
)

QA_CSV = (
    REPORTS
    / "16c_e_planning_evidence_qa.csv"
)

SUMMARY_JSON = (
    REPORTS
    / "16c_e_planning_evidence_summary.json"
)


# ============================================================
# 2. OFFICIAL PLANNING EVIDENCE REGISTRY
# ============================================================

# Page numbers were identified from the source document.
#
# pdf_page:
#     Page position in the PDF viewer (1-based).
#
# printed_page:
#     Page number printed inside the document.
#
# anchor:
#     Text marker used to confirm the PDF page.
#
# The anchor check does not independently verify
# the legal meaning of the source statement.

EVIDENCE = [

    {
        "id": "E01",
        "criterion": "C1-C5",
        "section": "7.1",
        "pdf_page": 204,
        "printed_page": 180,
        "anchor": "7.1",

        "finding": (
            "The zoning approach considers land use, "
            "population and building density, "
            "environmental sensitivity and "
            "proposed regulations."
        ),

        "implication": (
            "Supports using several types of planning "
            "evidence. Does not define GIS-MCDA weights."
        ),

        "use": "PLANNING_CONTEXT",
    },

    {
        "id": "E02",
        "criterion": "C1-C5",
        "section": "7.3",
        "pdf_page": 206,
        "printed_page": 182,
        "anchor": "7.3",

        "finding": (
            "The plan identifies seven main "
            "development-zone groups and subzones, "
            "including conservation zones."
        ),

        "implication": (
            "Use zoning in a separate spatial screening "
            "or comparison step, not as an input to "
            "independent core criteria."
        ),

        "use": "SEPARATE_CONSTRAINT_SCREEN",
    },

    {
        "id": "E03",
        "criterion": "C1-C5",
        "section": "7.4 (ii)",
        "pdf_page": 208,
        "printed_page": 184,
        "anchor": "7.4",

        "finding": (
            "The plan states that general UDA planning "
            "and building regulations published under "
            "Gazette 2235/54 also apply."
        ),

        "implication": (
            "The raw suitability score cannot be "
            "described as development permission."
        ),

        "use": "PLANNING_CONTEXT",
    },

    {
        "id": "E04",
        "criterion": "C4",
        "section": "7.4 (vii)",
        "pdf_page": 208,
        "printed_page": 184,
        "anchor": "7.4",

        "finding": (
            "Earth cutting on sloping land calls for "
            "relevant agency recommendations and a "
            "stated boundary protection measure."
        ),

        "implication": (
            "Slope has planning relevance, but this "
            "is NOT a universal slope-in-degrees "
            "suitability cutoff."
        ),

        "use": "PLANNING_CONTEXT",
    },

    {
        "id": "E05",
        "criterion": "C1-C2",
        "section": "7.4 (xi)",
        "pdf_page": 209,
        "printed_page": 185,
        "anchor": "xi.",

        "finding": (
            "Access-road width and UDA decision-making "
            "affect some proposals for development."
        ),

        "implication": (
            "OSM road proximity and density are access "
            "indicators, not proof of actual parcel access."
        ),

        "use": "PLANNING_CONTEXT",
    },

    {
        "id": "E06",
        "criterion": "C5",
        "section": "8.1.10",
        "pdf_page": 236,
        "printed_page": 212,
        "anchor": "8.1.10",

        "finding": (
            "Korathota highland conservation zone rules "
            "distinguish land up to 25 m above sea level."
        ),

        "implication": (
            "25 m is zone-specific; do NOT apply a 25 m "
            "elevation threshold to all Kaduwela grids."
        ),

        "use": "SEPARATE_CONSTRAINT_SCREEN",
    },

    {
        "id": "E07",
        "criterion": "C5",
        "section": "8.1.12-8.1.13",
        "pdf_page": 240,
        "printed_page": 216,
        "anchor": "8.1.13",

        "finding": (
            "Dedigamuwa highland conservation rules "
            "distinguish land at or below versus above "
            "30 m elevation. Also see printed page 215."
        ),

        "implication": (
            "30 m is zone-specific; do NOT apply a 30 m "
            "elevation threshold to all Kaduwela grids."
        ),

        "use": "SEPARATE_CONSTRAINT_SCREEN",
    },

    {
        "id": "E08",
        "criterion": "CONSERVATION",
        "section": "8.1.14",
        "pdf_page": 243,
        "printed_page": 219,
        "anchor": "8.1.14",

        "finding": (
            "Highly sensitive conservation areas include "
            "formally protected wetlands with restricted "
            "permitted uses."
        ),

        "implication": (
            "Use protected-area status as a later "
            "constraint. Not all permissible activities "
            "are ordinary construction."
        ),

        "use": "SEPARATE_CONSTRAINT_SCREEN",
    },

    {
        "id": "E09",
        "criterion": "CONSERVATION",
        "section": "8.1.16",
        "pdf_page": 250,
        "printed_page": 226,
        "anchor": "8.1.16",

        "finding": (
            "Low-sensitivity conservation zones also "
            "have defined permitted uses and conditions."
        ),

        "implication": (
            "Do not label every conservation zone as "
            "identical or automatically prohibit every use."
        ),

        "use": "SEPARATE_CONSTRAINT_SCREEN",
    },

    {
        "id": "E10",
        "criterion": "WETLANDS",
        "section": "Wetland special provisions",
        "pdf_page": 252,
        "printed_page": 228,
        "anchor": "Highland areas",

        "finding": (
            "Wetland-zone rules address uncertain "
            "boundaries, site surveys, drainage and "
            "flood-retention functions."
        ),

        "implication": (
            "Separate future wetland screening needs "
            "mapped boundaries and may require "
            "site-specific review. Do not use a universal "
            "20 m buffer."
        ),

        "use": "SEPARATE_CONSTRAINT_SCREEN",
    },

    {
        "id": "E11",
        "criterion": "C1",
        "section": "9.1",
        "pdf_page": 269,
        "printed_page": 242,
        "anchor": "9.1",

        "finding": (
            "Named roads have listed building lines of "
            "24 m, 15 m, 12 m or 9 m measured from "
            "the road centreline."
        ),

        "implication": (
            "These are named-road-specific requirements, "
            "not generic suitability-distance thresholds "
            "for every OSM road."
        ),

        "use": "SEPARATE_CONSTRAINT_SCREEN",
    },
]


# ============================================================
# 3. START
# ============================================================

print("=" * 92)
print("STEP 16C-E - KADUWELA DEVELOPMENT PLAN EVIDENCE AUDIT")
print("=" * 92)


# ============================================================
# 4. VERIFY REQUIRED INPUT FILES
# ============================================================

for path in [
    PDF,
    RAW_CSV,
    PREVIOUS,
]:

    if not path.is_file():

        raise FileNotFoundError(
            f"Required input not found: {path}"
        )


# ============================================================
# 5. LOAD PREVIOUS RESEARCH STATUS
# ============================================================

with PREVIOUS.open(
    encoding="utf-8"
) as file:

    previous = json.load(file)


# ============================================================
# 6. CALCULATE INPUT CHECKSUMS
# ============================================================

input_sha256 = hashlib.sha256(
    RAW_CSV.read_bytes()
).hexdigest()

pdf_sha256 = hashlib.sha256(
    PDF.read_bytes()
).hexdigest()


# ============================================================
# 7. QA CHECKS
# ============================================================

checks = []


def check(name, passed, detail):

    checks.append({
        "check": name,
        "passed": bool(passed),
        "detail": str(detail),
    })


check(
    "Previous 16C-D QA PASS",
    previous.get("qa_status") == "PASS",
    previous.get("qa_status"),
)

check(
    "Previous 16C-D remains provisional",
    previous.get("methodology_status") == "PROVISIONAL",
    previous.get("methodology_status"),
)

check(
    "Frozen criteria input unchanged",
    input_sha256 == previous.get("input_sha256"),
    "SHA256 match",
)


# ============================================================
# 8. CREATE PLANNING EVIDENCE DATAFRAME
# ============================================================

registry = pd.DataFrame(EVIDENCE)

check(
    "Evidence record IDs unique",
    registry["id"].is_unique,
    len(registry),
)

check(
    "All registry fields complete",
    (
        not registry.isna().any().any()
        and not registry.astype(str).eq("").any().any()
    ),
    "No blank fields",
)

check(
    "No unsupported automatic scoring",
    registry["use"].isin([
        "PLANNING_CONTEXT",
        "SEPARATE_CONSTRAINT_SCREEN",
    ]).all(),
    "Evidence is not a numerical suitability score",
)


# ============================================================
# 9. VERIFY PDF PAGE REFERENCES
# ============================================================

with fitz.open(PDF) as document:

    total_pdf_pages = len(document)

    check(
        "Planning PDF has full document length",
        total_pdf_pages >= 300,
        total_pdf_pages,
    )

    page_anchors = []

    print("\n=== PLANNING EVIDENCE PAGE CHECKS ===")

    for evidence in EVIDENCE:

        page_number = evidence["pdf_page"]

        if not 1 <= page_number <= total_pdf_pages:

            matched = False

        else:

            extracted_text = document[
                page_number - 1
            ].get_text("text")

            matched = (
                evidence["anchor"].casefold()
                in extracted_text.casefold()
            )

        page_anchors.append(matched)

        result = (
            "ANCHOR_OK"
            if matched
            else "CHECK_PAGE"
        )

        print(
            f"{evidence['id']} | "
            f"PDF p.{page_number:3d} | "
            f"Printed p.{evidence['printed_page']:3d} | "
            f"Section {evidence['section']} | "
            f"{result}"
        )

    check(
        "All cited PDF page anchors match",
        all(page_anchors),
        (
            f"{sum(page_anchors)}/"
            f"{len(page_anchors)} markers found"
        ),
    )


# ============================================================
# 10. ELEVATION RULE VALIDATION
# ============================================================

check(
    "Elevation restrictions remain zone-specific",
    all(
        "do NOT apply" in record["implication"]

        for record in EVIDENCE

        if record["id"] in [
            "E06",
            "E07",
        ]
    ),
    "No global 25 m / 30 m score cutoff",
)


# ============================================================
# 11. FINAL QA STATUS
# ============================================================

qa = pd.DataFrame(checks)

overall_pass = bool(
    qa["passed"].all()
)

status = (
    "PASS"
    if overall_pass
    else "REVIEW_REQUIRED"
)


# ============================================================
# 12. SAVE QA REPORT
# ============================================================

REPORTS.mkdir(
    parents=True,
    exist_ok=True,
)

qa.to_csv(
    QA_CSV,
    index=False,
)

if not overall_pass:

    print("\n=== QA FAILURE ===")

    print(
        qa.to_string(index=False)
    )

    raise RuntimeError(
        "Planning evidence QA failed. "
        "Existing evidence reports were not overwritten."
    )


# ============================================================
# 13. SAVE EVIDENCE REGISTRY
# ============================================================

registry.to_csv(
    REGISTRY_CSV,
    index=False,
)


# ============================================================
# 14. CREATE SUMMARY JSON
# ============================================================

summary = {
    "step": "16C-E",

    "qa_status": status,

    "methodology_status":
        "EVIDENCE_RECORDED_FINAL_CHOICES_PENDING",

    "generated_utc": datetime.now(
        timezone.utc
    ).isoformat(),

    "pdf_file": PDF.relative_to(
        ROOT
    ).as_posix(),

    "pdf_sha256": pdf_sha256,

    "pdf_pages": total_pdf_pages,

    "frozen_16b_csv_sha256": input_sha256,

    "evidence_items": len(EVIDENCE),

    "qa_total": len(qa),

    "qa_passed": int(
        qa["passed"].sum()
    ),

    "source_limitations": [

        (
            "Source statements were manually reviewed "
            "at the cited PDF pages. Automated QA checks "
            "page anchors, not legal meaning."
        ),

        (
            "Sinhala PDF extraction may be imperfect. "
            "Confirm exact legal wording in the PDF."
        ),

        (
            "Document-specific zoning rules are not "
            "city-wide GIS-MCDA thresholds."
        ),

        (
            "Official development permission requires "
            "applicable agency procedures and "
            "site-specific review."
        ),

        (
            "UDA provisional zoning GIS is not used "
            "to calculate raw C1-C5 criteria."
        ),

        (
            "C2/C3 500m proposals, C3 direction, C5 "
            "inclusion and standardization thresholds "
            "remain provisional."
        ),
    ],

    "outputs": [
        REGISTRY_CSV.name,
        QA_CSV.name,
        SUMMARY_JSON.name,
    ],
}


# ============================================================
# 15. SAVE SUMMARY
# ============================================================

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
# 16. TERMINAL REPORT
# ============================================================

print("\n=== STEP 16C-E QA RESULTS ===")

for record in checks:

    result = (
        "PASS"
        if record["passed"]
        else "FAIL"
    )

    print(
        f"{result} : {record['check']}"
    )


print("\n=== PLANNING EVIDENCE SUMMARY ===")

print(
    "Total evidence records:",
    len(registry),
)

print(
    "Total PDF pages:",
    total_pdf_pages,
)

print(
    "QA status:",
    status,
)

print(
    "Research decisions:",
    "STILL PROVISIONAL",
)


print("\n=== GENERATED REPORTS ===")

for path in [
    REGISTRY_CSV,
    QA_CSV,
    SUMMARY_JSON,
]:

    print(
        " -",
        path.relative_to(ROOT),
    )


print("\nIMPORTANT:")

print(
    "No normalization, AHP weighting "
    "or suitability scoring was performed."
)

print(
    "The PDF page checks do not independently "
    "verify the legal meaning of the planning rules."
)

print("=" * 92)
