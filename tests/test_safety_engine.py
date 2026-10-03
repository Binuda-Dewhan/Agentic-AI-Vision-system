import pytest

from sleep_monitor.config.settings import SafetyRulesConfig
from sleep_monitor.engine.safety_engine import SafetyEngine
from sleep_monitor.engine.timeline_engine import TimelineEngine
from sleep_monitor.schemas.events import BedEvent
from sleep_monitor.schemas.state import ActivityState, BedContext, SafetyDecision


@pytest.fixture
def safety_config():
    return SafetyRulesConfig(
        out_of_bed_monitor_sec=300,  # 5 min
        alert_out_of_bed_duration_sec=900,  # 15 min
        max_exits_before_alert=3,
    )


def test_safety_normal(safety_config):
    engine = SafetyEngine(safety_config)
    timeline = TimelineEngine()

    dec = engine.evaluate(ActivityState.LYING_IN_BED, BedContext.IN_BED, timeline)
    assert dec == SafetyDecision.NORMAL


def test_safety_monitor_duration(safety_config):
    engine = SafetyEngine(safety_config)
    timeline = TimelineEngine()

    # Simulate being out of bed for 6 minutes (360s)
    timeline.current_out_duration = 360
    dec = engine.evaluate(ActivityState.STANDING, BedContext.OUT_OF_BED, timeline)
    assert dec == SafetyDecision.MONITOR

    # Now they return to bed
    dec2 = engine.evaluate(ActivityState.LYING_IN_BED, BedContext.IN_BED, timeline)
    assert dec2 == SafetyDecision.NORMAL  # Recovered


def test_safety_alert_duration(safety_config):
    engine = SafetyEngine(safety_config)
    timeline = TimelineEngine()

    # Simulate being out of bed for 16 minutes (960s)
    timeline.current_out_duration = 960
    dec = engine.evaluate(ActivityState.STANDING, BedContext.OUT_OF_BED, timeline)
    assert dec == SafetyDecision.ALERT


def test_safety_alert_fall(safety_config):
    engine = SafetyEngine(safety_config)
    timeline = TimelineEngine()

    # Simulate lying on the floor (LYING_IN_BED posture but OUT_OF_BED context)
    dec = engine.evaluate(ActivityState.LYING_IN_BED, BedContext.OUT_OF_BED, timeline)
    assert dec == SafetyDecision.ALERT


def test_safety_multiple_exits(safety_config):
    engine = SafetyEngine(safety_config)
    timeline = TimelineEngine()

    # Add 4 exit events
    for _ in range(4):
        timeline.bed_events.append(
            BedEvent(event_type="BED_EXIT", timestamp_sec=0, confirmed_at_sec=0)
        )

    dec = engine.evaluate(ActivityState.LYING_IN_BED, BedContext.IN_BED, timeline)
    assert (
        dec == SafetyDecision.ALERT
    )  # Alert persists even if in bed due to restless history
