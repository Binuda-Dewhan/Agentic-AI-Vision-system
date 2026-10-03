import logging

from sleep_monitor.schemas.events import BedEvent
from sleep_monitor.schemas.report import (
    BedSummary,
    FinalReport,
    TargetPersonInfo,
    TimelineSegment,
)
from sleep_monitor.schemas.state import ActivityState, BedContext

logger = logging.getLogger(__name__)


class TimelineEngine:
    """
    Aggregates frame-by-frame or segment-by-segment state changes into
    the final duration, timeline, and bed summary outputs.
    """

    def __init__(self):
        self.timeline: list[TimelineSegment] = []
        self.current_segment: TimelineSegment | None = None

        # Track activity durations
        # NOTE: OUT_OF_BED is a bed context, not an activity. It is tracked
        # separately in bed_summary.time_out_of_bed to avoid double-counting
        # (e.g., WALKING while OUT_OF_BED would count in both otherwise).
        self.activity_durations: dict[str, float] = {
            state.value: 0.0 for state in ActivityState
        }

        self.bed_events: list[BedEvent] = []

        # Bed summary tracking
        self.time_in_bed = 0.0
        self.time_out_of_bed = 0.0
        self.current_out_duration = 0.0
        self.longest_out_duration = 0.0

    def process_segment(
        self,
        start_time: float,
        end_time: float,
        activity: ActivityState,
        bed_context: BedContext,
        event: BedEvent | None,
    ):
        """Process a processed temporal segment (e.g. 2 seconds)."""
        duration = end_time - start_time
        if duration <= 0:
            return

        # 1. Timeline Generation (Merge contiguous states)
        if self.current_segment is None:
            self.current_segment = TimelineSegment(
                start_time_sec=start_time, end_time_sec=end_time, state=activity.value
            )
        elif self.current_segment.state == activity.value:
            # Extend current segment if state hasn't changed
            self.current_segment.end_time_sec = end_time
        else:
            # State changed: complete current segment and start new
            self.timeline.append(self.current_segment)
            self.current_segment = TimelineSegment(
                start_time_sec=start_time, end_time_sec=end_time, state=activity.value
            )

        # 2. Activity Durations
        self.activity_durations[activity.value] += duration

        # 3. Bed Context & Durations
        if bed_context == BedContext.IN_BED:
            self.time_in_bed += duration

            # Reset current out duration if returning to bed
            if self.current_out_duration > 0:
                self.longest_out_duration = max(
                    self.longest_out_duration, self.current_out_duration
                )
                self.current_out_duration = 0.0

        elif bed_context == BedContext.OUT_OF_BED:
            self.time_out_of_bed += duration
            self.current_out_duration += duration

        # 4. Record Events
        if event:
            self.bed_events.append(event)

    def finalize(self) -> FinalReport:
        """Complete the processing and return the FinalReport."""
        # Complete the last timeline segment
        if self.current_segment is not None:
            self.timeline.append(self.current_segment)

        # Check if the final state was out of bed, to update longest duration
        if self.current_out_duration > 0:
            self.longest_out_duration = max(
                self.longest_out_duration, self.current_out_duration
            )

        final_state = (
            self.timeline[-1].state if self.timeline else ActivityState.UNKNOWN.value
        )

        exit_count = sum(1 for e in self.bed_events if e.event_type == "BED_EXIT")
        return_count = sum(
            1 for e in self.bed_events if e.event_type == "RETURN_TO_BED"
        )

        # Clean up empty durations for a cleaner report
        clean_durations = {k: v for k, v in self.activity_durations.items() if v > 0}

        return FinalReport(
            target_person=TargetPersonInfo(final_state=final_state),
            activity_durations=clean_durations,
            bed_summary=BedSummary(
                time_in_bed=self.time_in_bed,
                time_out_of_bed=self.time_out_of_bed,
                exit_count=exit_count,
                return_count=return_count,
                longest_out_of_bed=self.longest_out_duration,
            ),
            bed_events=self.bed_events,
            timeline=self.timeline,
            # Safety decision will be set by the SafetyEngine later
        )
