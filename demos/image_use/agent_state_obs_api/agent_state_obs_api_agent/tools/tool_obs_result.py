from enum import Enum

from langchain.tools import tool
from pydantic import BaseModel, Field


class TriValue(str, Enum):
    TRUE = "TRUE"
    FALSE = "FALSE"
    UNKNOWN = "UNKNOWN"


class FailureMode(str, Enum):
    # Used when value is TRUE or FALSE and the model is confident
    CONFIDENT = "CONFIDENT"

    # Uncertainty / visibility issues
    OCCLUDED = "OCCLUDED"
    REFERENTS_NOT_VISIBLE = "REFERENTS_NOT_VISIBLE"
    LOW_RESOLUTION = "LOW_RESOLUTION"
    AMBIGUOUS = "AMBIGUOUS"
    MODEL_UNCERTAIN = "MODEL_UNCERTAIN"

    # Input issues
    BAD_IMAGE = "BAD_IMAGE"

    # Catch-all for anything not covered
    OTHER = "OTHER"


class ObsResult(BaseModel):
    """
    Minimal structured observation result produced by the VLM tool.

    Rules:
    - If failure_mode == CONFIDENT, value should be TRUE or FALSE.
    - If value == UNKNOWN, failure_mode should not be CONFIDENT.
    """
    value: TriValue = Field(
        ...,
        description='Ternary decision: "TRUE", "FALSE", or "UNKNOWN".',
    )

    failure_mode: FailureMode = Field(
        ...,
        description="Categorized reason for the decision or uncertainty.",
    )

    reason: str = Field(
        ...,
        description="Short explanation (1–2 sentences) justifying the decision.",
        min_length=1,
        max_length=600,
    )

    model_config = {
        "extra": "forbid"
    }


class ObsResultArgs(BaseModel):
    value: TriValue = Field(
        ...,
        description='Ternary decision: "TRUE", "FALSE", or "UNKNOWN".',
    )
    failure_mode: FailureMode = Field(
        ...,
        description="Categorized reason for the decision or uncertainty.",
    )
    reason: str = Field(
        ...,
        description="Short explanation (1-2 sentences) justifying the decision.",
        min_length=1,
        max_length=600,
    )


@tool(args_schema=ObsResultArgs)
def obs_result_tool(value: TriValue, failure_mode: FailureMode, reason: str) -> str:
    """Record a structured observation result. Always use this tool to provide your response."""
    return ObsResult(
        value=value,
        failure_mode=failure_mode,
        reason=reason,
    ).model_dump_json()
