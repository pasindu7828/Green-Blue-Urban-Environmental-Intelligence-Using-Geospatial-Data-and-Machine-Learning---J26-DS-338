"""Pydantic request schemas for Urban Heat dynamic scenario APIs."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field, model_validator


class ScenarioFamily(str, Enum):
    GREEN = "GREEN"
    BUILT = "BUILT"
    BLUE = "BLUE"
    COMBINED = "COMBINED"


class UrbanHeatScenarioRequest(BaseModel):
    """One-grid, one-target-year dynamic planning scenario request."""

    grid_id: str = Field(..., min_length=1, examples=["1832,3615"])
    year: int = Field(..., ge=2026, le=2030, examples=[2030])
    scenario_family: ScenarioFamily

    green_increase_pp: float | None = Field(
        default=None,
        ge=0.0,
        le=20.0,
        description=(
            "Persistent green-cover increase in percentage points. Required for "
            "GREEN and COMBINED. Research levels: 5, 10, 20 pp."
        ),
        examples=[10.0],
    )
    ndbi_reduction: float | None = Field(
        default=None,
        ge=0.0,
        description=(
            "Persistent NDBI reduction. Required for BUILT and COMBINED. "
            "The service enforces the research-derived upper envelope."
        ),
        examples=[0.039658563922391525],
    )
    ndwi_increase: float | None = Field(
        default=None,
        ge=0.0,
        description=(
            "Persistent NDWI increase. Required for BLUE. The service enforces "
            "the research-derived upper envelope."
        ),
        examples=[0.0666914129945183],
    )

    @model_validator(mode="after")
    def validate_family_inputs(self) -> "UrbanHeatScenarioRequest":
        family = self.scenario_family

        if family is ScenarioFamily.GREEN and self.green_increase_pp is None:
            raise ValueError("green_increase_pp is required for GREEN.")

        if family is ScenarioFamily.BUILT and self.ndbi_reduction is None:
            raise ValueError("ndbi_reduction is required for BUILT.")

        if family is ScenarioFamily.BLUE and self.ndwi_increase is None:
            raise ValueError("ndwi_increase is required for BLUE.")

        if family is ScenarioFamily.COMBINED:
            if self.green_increase_pp is None:
                raise ValueError("green_increase_pp is required for COMBINED.")
            if self.ndbi_reduction is None:
                raise ValueError("ndbi_reduction is required for COMBINED.")

        return self
