from pydantic import BaseModel, Field

from sleep_monitor.schemas.state import ActivityState


class AgentDecision(BaseModel):
    """The final decision made by the investigation agent."""

    confirmed_state: ActivityState = Field(
        description="The resolved ActivityState. Choose UNKNOWN if it remains ambiguous."
    )
    confidence: float = Field(
        description="Confidence in the decision, from 0.0 to 1.0."
    )
    decision_type: str = Field(
        default="UNABLE_TO_RESOLVE",
        description="Type of decision: CONFIRM_STATE, CONFIRM_BED_EXIT, CONFIRM_RETURN, UNABLE_TO_RESOLVE",
    )
    reasoning: str = Field(
        default="", description="Explanation of how the decision was reached."
    )
    evidence: list[str] = Field(
        default_factory=list, description="Reasons supporting the decision."
    )
    uncertainty: list[str] = Field(
        default_factory=list, description="Remaining ambiguities."
    )
    vlm_used: bool = Field(
        default=False,
        description="Whether the VLM was invoked during this investigation.",
    )
