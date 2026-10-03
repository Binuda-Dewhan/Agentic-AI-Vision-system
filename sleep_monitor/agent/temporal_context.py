"""Schemas used by the investigation agent for temporal reasoning."""

from pydantic import BaseModel, Field

from sleep_monitor.schemas.perception import ActivityEvidence, Observation
from sleep_monitor.schemas.state import ActivityState, BedContext


class StateHistoryEntry(BaseModel):
    """A single entry in the recent state history."""

    time_sec: float
    activity: str
    bed_context: str


class TemporalContext(BaseModel):
    """
    Rich temporal context passed to the investigation agent.

    Contains previous/current/next segment summaries plus recent state history,
    enabling the agent to reason about temporal transitions rather than
    single-frame classification.

    The 'next' segment is only populated when a buffered look-ahead is
    available — it is never fabricated.
    """

    # Current segment (always present)
    current_segment_id: int = 0
    current_start_sec: float = 0.0
    current_end_sec: float = 0.0
    current_evidence: ActivityEvidence = Field(default_factory=ActivityEvidence)
    current_spatial: str = "UNKNOWN"

    # Previous segment (if available)
    previous_evidence: ActivityEvidence | None = None
    previous_spatial: str | None = None
    previous_activity: str | None = None

    # Next / following segment (only when buffered look-ahead exists)
    next_evidence: ActivityEvidence | None = None
    next_spatial: str | None = None

    # Recent state history (last N transitions)
    state_history: list[StateHistoryEntry] = Field(default_factory=list)

    # Current bed context
    bed_context: str = "UNSET"

    # Recent bed event candidates (pending exit/return)
    pending_exit_since_sec: float | None = None
    pending_return_since_sec: float | None = None

    def format_for_prompt(self) -> str:
        """Format temporal context as a human-readable string for the LLM prompt."""
        lines = []

        lines.append(
            f"=== Current Segment {self.current_segment_id} "
            f"({self.current_start_sec:.1f}s - {self.current_end_sec:.1f}s) ==="
        )
        lines.append(
            f"Evidence: lying={self.current_evidence.lying:.2f}, "
            f"sitting={self.current_evidence.sitting:.2f}, "
            f"standing={self.current_evidence.standing:.2f}, "
            f"walking={self.current_evidence.walking:.2f}"
        )
        lines.append(f"Spatial Position: {self.current_spatial}")

        if self.previous_evidence:
            lines.append("\n--- Previous Segment ---")
            lines.append(
                f"Evidence: lying={self.previous_evidence.lying:.2f}, "
                f"sitting={self.previous_evidence.sitting:.2f}, "
                f"standing={self.previous_evidence.standing:.2f}, "
                f"walking={self.previous_evidence.walking:.2f}"
            )
            lines.append(f"Spatial: {self.previous_spatial}")
            lines.append(f"Determined Activity: {self.previous_activity}")

        if self.next_evidence:
            lines.append("\n--- Following Segment (look-ahead) ---")
            lines.append(
                f"Evidence: lying={self.next_evidence.lying:.2f}, "
                f"sitting={self.next_evidence.sitting:.2f}, "
                f"standing={self.next_evidence.standing:.2f}, "
                f"walking={self.next_evidence.walking:.2f}"
            )
            lines.append(f"Spatial: {self.next_spatial}")

        if self.state_history:
            lines.append(f"\n--- Recent State History (last {len(self.state_history)}) ---")
            for entry in self.state_history:
                lines.append(
                    f"  t={entry.time_sec:.1f}s: {entry.activity} [{entry.bed_context}]"
                )

        lines.append(f"\nCurrent Bed Context: {self.bed_context}")

        if self.pending_exit_since_sec is not None:
            lines.append(
                f"⚠ Pending BED_EXIT since {self.pending_exit_since_sec:.1f}s"
            )
        if self.pending_return_since_sec is not None:
            lines.append(
                f"⚠ Pending RETURN_TO_BED since {self.pending_return_since_sec:.1f}s"
            )

        return "\n".join(lines)
