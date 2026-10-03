import pytest

from sleep_monitor.config.settings import BedEventsConfig
from sleep_monitor.engine.event_engine import BedEventEngine
from sleep_monitor.schemas.perception import ActivityEvidence, Observation
from sleep_monitor.schemas.state import ActivityState, BedContext


@pytest.fixture
def event_config():
    return BedEventsConfig(bed_exit_hysteresis_sec=10, return_hysteresis_sec=5)


def create_mock_obs(start_time, end_time, spatial):
    return Observation(
        segment_id=int(start_time),
        start_time_sec=start_time,
        end_time_sec=end_time,
        frames=[],
        other_persons_present=False,
        average_evidence=ActivityEvidence(),
        majority_spatial_position=spatial,
    )


def test_bed_exit_success(event_config):
    engine = BedEventEngine(event_config)
    context = BedContext.IN_BED

    # 0s to 5s: Standing outside bed (potential exit starts)
    obs1 = create_mock_obs(0.0, 5.0, "OUTSIDE")
    context, event = engine.process(ActivityState.STANDING, context, obs1)

    assert event is None
    assert context == BedContext.IN_BED
    assert engine.potential_exit_start_time == 0.0

    # 5s to 12s: Still standing outside bed (hysteresis 10s passed)
    obs2 = create_mock_obs(5.0, 12.0, "OUTSIDE")
    context, event = engine.process(ActivityState.STANDING, context, obs2)

    assert event is not None
    assert event.event_type == "BED_EXIT"
    assert event.timestamp_sec == 0.0  # Time when they first stood up
    assert event.confirmed_at_sec == 12.0
    assert context == BedContext.OUT_OF_BED


def test_bed_exit_false_alarm(event_config):
    engine = BedEventEngine(event_config)
    context = BedContext.IN_BED

    # 0s to 5s: Standing outside bed (potential exit starts)
    obs1 = create_mock_obs(0.0, 5.0, "OUTSIDE")
    engine.process(ActivityState.STANDING, context, obs1)
    assert engine.potential_exit_start_time == 0.0

    # 5s to 8s: Sits back on bed (false exit prevented)
    obs2 = create_mock_obs(5.0, 8.0, "INSIDE")
    context, event = engine.process(ActivityState.SITTING_ON_BED, context, obs2)

    assert event is None
    assert context == BedContext.IN_BED
    assert engine.potential_exit_start_time is None  # Aborted


def test_return_to_bed(event_config):
    engine = BedEventEngine(event_config)
    context = BedContext.OUT_OF_BED

    # 0s to 3s: Sits on bed (potential return starts)
    obs1 = create_mock_obs(0.0, 3.0, "ON_EDGE")
    context, event = engine.process(ActivityState.SITTING_ON_BED, context, obs1)

    assert event is None
    assert context == BedContext.OUT_OF_BED
    assert engine.potential_return_start_time == 0.0

    # 3s to 6s: Lying in bed (hysteresis 5s passed)
    obs2 = create_mock_obs(3.0, 6.0, "INSIDE")
    context, event = engine.process(ActivityState.LYING_IN_BED, context, obs2)

    assert event is not None
    assert event.event_type == "RETURN_TO_BED"
    assert event.timestamp_sec == 0.0
    assert context == BedContext.IN_BED
