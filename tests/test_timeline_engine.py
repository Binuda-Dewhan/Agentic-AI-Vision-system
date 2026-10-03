from sleep_monitor.engine.timeline_engine import TimelineEngine
from sleep_monitor.schemas.events import BedEvent
from sleep_monitor.schemas.state import ActivityState, BedContext


def test_timeline_engine():
    engine = TimelineEngine()

    # Segment 1: Lying in bed (2s)
    engine.process_segment(
        0.0, 2.0, ActivityState.LYING_IN_BED, BedContext.IN_BED, None
    )

    # Segment 2: Sitting on bed (2s)
    engine.process_segment(
        2.0, 4.0, ActivityState.SITTING_ON_BED, BedContext.IN_BED, None
    )

    # Segment 3: Standing outside (2s, triggers exit)
    event_exit = BedEvent(
        event_type="BED_EXIT", timestamp_sec=4.0, confirmed_at_sec=6.0
    )
    engine.process_segment(
        4.0, 6.0, ActivityState.STANDING, BedContext.OUT_OF_BED, event_exit
    )

    # Segment 4: Walking outside (4s)
    engine.process_segment(
        6.0, 10.0, ActivityState.WALKING, BedContext.OUT_OF_BED, None
    )

    # Segment 5: Walking outside (2s) -> should merge with previous WALKING segment
    engine.process_segment(
        10.0, 12.0, ActivityState.WALKING, BedContext.OUT_OF_BED, None
    )

    # Finalize
    report = engine.finalize()

    # Validate final state
    assert report.target_person.final_state == ActivityState.WALKING.value

    # Validate timeline merging
    assert len(report.timeline) == 4
    assert report.timeline[0].state == ActivityState.LYING_IN_BED.value
    assert report.timeline[0].end_time_sec == 2.0

    assert report.timeline[-1].state == ActivityState.WALKING.value
    assert report.timeline[-1].start_time_sec == 6.0
    assert report.timeline[-1].end_time_sec == 12.0

    # Validate Activity Durations
    assert report.activity_durations[ActivityState.LYING_IN_BED.value] == 2.0
    assert report.activity_durations[ActivityState.SITTING_ON_BED.value] == 2.0
    assert report.activity_durations[ActivityState.STANDING.value] == 2.0
    assert report.activity_durations[ActivityState.WALKING.value] == 6.0
    assert report.activity_durations["OUT_OF_BED"] == 8.0  # 2s standing + 6s walking

    # Validate Bed Summary
    assert report.bed_summary.time_in_bed == 4.0
    assert report.bed_summary.time_out_of_bed == 8.0
    assert report.bed_summary.exit_count == 1
    assert report.bed_summary.return_count == 0
    assert report.bed_summary.longest_out_of_bed == 8.0

    # Validate Events
    assert len(report.bed_events) == 1
    assert report.bed_events[0].event_type == "BED_EXIT"
