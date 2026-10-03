from pydantic import BaseModel

from sleep_monitor.schemas.events import BedEvent


class TimelineSegment(BaseModel):
    """A contiguous period of time spent in a single activity state."""

    start_time_sec: float
    end_time_sec: float
    state: str


class TargetPersonInfo(BaseModel):
    final_state: str


class BedSummary(BaseModel):
    time_in_bed: float
    time_out_of_bed: float
    exit_count: int
    return_count: int
    longest_out_of_bed: float


class FinalReport(BaseModel):
    """The final structured output matching the assignment requirements."""

    observation_duration_sec: float = 0.0
    target_person: TargetPersonInfo
    activity_durations: dict[str, float]
    bed_summary: BedSummary
    bed_events: list[BedEvent]
    timeline: list[TimelineSegment]
    safety_decision: str = "NORMAL"
