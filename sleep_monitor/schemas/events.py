from pydantic import BaseModel


class BedEvent(BaseModel):
    """Represents a confirmed bed exit or return to bed event."""

    event_type: str  # "BED_EXIT" or "RETURN_TO_BED"
    timestamp_sec: float
    confirmed_at_sec: float
    previous_state: str = ""
    current_state: str = ""
    confidence: float = 0.0
    decision: str = "NORMAL"  # Safety decision at time of event
