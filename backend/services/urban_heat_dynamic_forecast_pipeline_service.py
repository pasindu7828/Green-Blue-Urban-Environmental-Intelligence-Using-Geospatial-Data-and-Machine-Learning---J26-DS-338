from __future__ import annotations

from datetime import datetime
from pathlib import Path
import shutil
import json
import subprocess
import threading
import time

from backend.config import URBAN_HEAT_ROOT


DYNAMIC_FORECAST_ROOT = (
    URBAN_HEAT_ROOT
    / "experiments"
    / "dynamic_5year_forecast"
)

UPLOAD_DIR = (
    DYNAMIC_FORECAST_ROOT
    / "uploads"
)

ARCHIVE_DIR = (
    UPLOAD_DIR
    / "archive"
)

DYNAMIC_SOURCE_DIR = (
    DYNAMIC_FORECAST_ROOT
    / "src"
)

STEP01_SCRIPT = (
    DYNAMIC_SOURCE_DIR
    / "01_validate_uploaded_history.py"
)

STEP01_SUMMARY_FILE = (
    DYNAMIC_FORECAST_ROOT
    / "outputs"
    / "reports"
    / "01_upload_validation_summary.json"
)

RESEARCH_PYTHON = (
    URBAN_HEAT_ROOT
    / ".venv"
    / "Scripts"
    / "python.exe"
)

STEP02_SCRIPT = (
    DYNAMIC_SOURCE_DIR
    / "02_refresh_trajectory.py"
)

STEP03_SCRIPT = (
    DYNAMIC_SOURCE_DIR
    / "03_retrain_dynamic_spatial_model.py"
)

STEP04_SCRIPT = (
    DYNAMIC_SOURCE_DIR
    / "04_project_dynamic_future_inputs.py"
)

STEP05_SCRIPT = (
    DYNAMIC_SOURCE_DIR
    / "05_generate_dynamic_5year_forecast.py"
)

STEP06_SCRIPT = (
    DYNAMIC_SOURCE_DIR
    / "06_final_dynamic_forecast_audit.py"
)

STEP06_MANIFEST_FILE = (
    DYNAMIC_FORECAST_ROOT
    / "outputs"
    / "reports"
    / "06_dynamic_service_manifest.json"
)

PIPELINE_STATE_FILE = (
    DYNAMIC_FORECAST_ROOT
    / "outputs"
    / "reports"
    / "backend_dynamic_pipeline_state.json"
)

PIPELINE_LOCK = threading.Lock()

ACTIVE_UPLOAD_NAME = "current_history_upload.csv"


def stage_dynamic_history_upload(
    source_file,
    original_filename: str,
) -> dict:

    if not original_filename:
        raise ValueError(
            "Uploaded file must have a filename."
        )

    if not original_filename.lower().endswith(".csv"):
        raise ValueError(
            "Only CSV files are accepted."
        )

    UPLOAD_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    ARCHIVE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Archive previous CSV files so Step 01 sees exactly one active CSV.
    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    archived_files = []

    for existing in UPLOAD_DIR.glob("*.csv"):

        archive_path = (
            ARCHIVE_DIR
            / f"{timestamp}_{existing.name}"
        )

        shutil.move(
            str(existing),
            str(archive_path),
        )

        archived_files.append(
            str(archive_path)
        )

    destination = (
        UPLOAD_DIR
        / ACTIVE_UPLOAD_NAME
    )

    with destination.open(
        "wb"
    ) as output_file:

        shutil.copyfileobj(
            source_file,
            output_file,
        )

    size_bytes = destination.stat().st_size

    if size_bytes == 0:

        destination.unlink(
            missing_ok=True
        )

        raise ValueError(
            "Uploaded CSV is empty."
        )

    return {
        "status": "uploaded",
        "original_filename": original_filename,
        "active_filename": ACTIVE_UPLOAD_NAME,
        "saved_path": str(destination),
        "size_bytes": size_bytes,
        "archived_previous_csv_count": len(
            archived_files
        ),
        "message": (
            "CSV staged successfully. "
            "Research pipeline has not been executed yet."
        ),
    }



def validate_staged_dynamic_history() -> dict:

    active_csv = (
        UPLOAD_DIR
        / ACTIVE_UPLOAD_NAME
    )

    if not active_csv.exists():

        raise FileNotFoundError(
            "No staged historical CSV was found. "
            "Upload a CSV first."
        )

    if not STEP01_SCRIPT.exists():

        raise FileNotFoundError(
            f"Step 01 validation script was not found: "
            f"{STEP01_SCRIPT}"
        )

    if not RESEARCH_PYTHON.exists():

        raise FileNotFoundError(
            f"Urban Heat research Python environment "
            f"was not found: {RESEARCH_PYTHON}"
        )

    result = subprocess.run(
        [
            str(RESEARCH_PYTHON),
            str(STEP01_SCRIPT),
        ],
        cwd=str(
            URBAN_HEAT_ROOT
        ),
        capture_output=True,
        text=True,
        timeout=300,
    )

    if result.returncode != 0:

        raise RuntimeError(
            "Uploaded CSV validation failed.\n\n"
            f"STDOUT:\n{result.stdout}\n\n"
            f"STDERR:\n{result.stderr}"
        )

    if not STEP01_SUMMARY_FILE.exists():

        raise RuntimeError(
            "Step 01 completed but validation summary "
            "was not generated."
        )

    with STEP01_SUMMARY_FILE.open(
        "r",
        encoding="utf-8",
    ) as file:

        summary = json.load(
            file
        )

    return {
        "status": "validated",
        "pipeline_step": 1,
        "active_upload": str(
            active_csv
        ),
        "latest_observed_year":
            summary.get(
                "latest_observed_year"
            ),

        "forecast_years":
            summary.get(
                "forecast_years"
            ),

        "validation_summary":
            summary,

        "message": (
            "Uploaded historical CSV passed "
            "Step 01 validation. "
            "No model update or future forecast "
            "has been executed yet."
        ),
    }

def _write_pipeline_state(
    status: str,
    current_step: str | None = None,
    message: str | None = None,
) -> None:

    PIPELINE_STATE_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    payload = {
        "status": status,
        "current_step": current_step,
        "message": message,
        "updated_at": datetime.now().isoformat(
            timespec="seconds"
        ),
    }

    with PIPELINE_STATE_FILE.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            payload,
            file,
            indent=2,
        )


def get_dynamic_pipeline_state() -> dict:

    if not PIPELINE_STATE_FILE.exists():

        return {
            "status": "idle",
            "current_step": None,
            "message": (
                "Dynamic forecast pipeline "
                "has not been started by the backend."
            ),
        }

    with PIPELINE_STATE_FILE.open(
        "r",
        encoding="utf-8",
    ) as file:

        return json.load(file)


def run_dynamic_forecast_pipeline() -> dict:

    if not PIPELINE_LOCK.acquire(
        blocking=False
    ):

        raise RuntimeError(
            "Dynamic forecast pipeline is already running."
        )

    try:

        if not STEP01_SUMMARY_FILE.exists():

            raise RuntimeError(
                "Step 01 validation output was not found. "
                "Validate the uploaded CSV first."
            )

        with STEP01_SUMMARY_FILE.open(
            "r",
            encoding="utf-8",
        ) as file:

            step01_summary = json.load(
                file
            )

        latest_year = step01_summary.get(
            "latest_observed_year"
        )

        forecast_years = step01_summary.get(
            "forecast_years"
        )

        scripts = [
            (
                "02_refresh_trajectory",
                STEP02_SCRIPT,
            ),
            (
                "03_controlled_model_update",
                STEP03_SCRIPT,
            ),
            (
                "04_future_input_projection",
                STEP04_SCRIPT,
            ),
            (
                "05_grid_lst_forecast",
                STEP05_SCRIPT,
            ),
            (
                "06_final_service_audit",
                STEP06_SCRIPT,
            ),
        ]

        step_results = []

        _write_pipeline_state(
            status="running",
            current_step="starting",
            message=(
                "Dynamic forecast pipeline started."
            ),
        )

        for step_name, script_path in scripts:

            if not script_path.exists():

                raise FileNotFoundError(
                    f"Pipeline script not found: "
                    f"{script_path}"
                )

            _write_pipeline_state(
                status="running",
                current_step=step_name,
                message=(
                    f"Running {step_name}."
                ),
            )

            started = time.perf_counter()

            result = subprocess.run(
                [
                    str(RESEARCH_PYTHON),
                    str(script_path),
                ],
                cwd=str(
                    URBAN_HEAT_ROOT
                ),
                capture_output=True,
                text=True,
                timeout=3600,
            )

            elapsed_seconds = round(
                time.perf_counter()
                - started,
                2,
            )

            stdout_lines = (
                result.stdout
                .strip()
                .splitlines()
            )

            stderr_lines = (
                result.stderr
                .strip()
                .splitlines()
            )

            step_result = {
                "step": step_name,
                "return_code":
                    result.returncode,

                "elapsed_seconds":
                    elapsed_seconds,

                "stdout_tail":
                    stdout_lines[-12:],

                "stderr_tail":
                    stderr_lines[-12:],
            }

            step_results.append(
                step_result
            )

            if result.returncode != 0:

                _write_pipeline_state(
                    status="failed",
                    current_step=step_name,
                    message=(
                        f"{step_name} failed."
                    ),
                )

                raise RuntimeError(
                    f"{step_name} failed.\n\n"
                    f"STDOUT:\n{result.stdout}\n\n"
                    f"STDERR:\n{result.stderr}"
                )

        if not STEP06_MANIFEST_FILE.exists():

            raise RuntimeError(
                "Step 06 completed but final dynamic "
                "service manifest was not found."
            )

        with STEP06_MANIFEST_FILE.open(
            "r",
            encoding="utf-8",
        ) as file:

            final_manifest = json.load(
                file
            )

        module_status = final_manifest.get(
            "status"
        )

        if module_status != "READY_FOR_API_INTEGRATION":

            _write_pipeline_state(
                status="failed",
                current_step="06_final_service_audit",
                message=(
                    "Final audit did not mark the "
                    "module ready."
                ),
            )

            raise RuntimeError(
                "Dynamic forecast pipeline finished, "
                "but Step 06 status was "
                f"'{module_status}'."
            )

        _write_pipeline_state(
            status="completed",
            current_step=None,
            message=(
                "Dynamic forecast pipeline "
                "completed successfully."
            ),
        )

        return {
            "status": "completed",
            "latest_observed_year":
                latest_year,

            "forecast_years":
                forecast_years,

            "module_status":
                module_status,

            "steps":
                step_results,

            "message": (
                "Dynamic forecast pipeline "
                "completed successfully."
            ),
        }

    finally:

        PIPELINE_LOCK.release()

