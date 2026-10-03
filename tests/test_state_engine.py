import pytest

from sleep_monitor.config.settings import TemporalStateConfig
from sleep_monitor.engine.state_engine import StateEngine
from sleep_monitor.schemas.perception import (
    ActivityEvidence,
    FrameObservation,
    Observation,
)
from sleep_monitor.schemas.state import ActivityState, BedContext


@pytest.fixture
def state_config():
    return TemporalStateConfig(
        state_confirmation_window_sec=4,
        state_transition_hysteresis=1.5,
        confidence_threshold=0.5,
    )


def create_mock_obs(start_time, end_time, evidence, spatial="INSIDE", valid_frames=1):
    frames = [
        FrameObservation(
            frame_index=0,
            timestamp_sec=start_time,
            bbox={"x1": 0, "y1": 0, "x2": 10, "y2": 10},
        )
    ] * valid_frames

    return Observation(
        segment_id=int(start_time),
        start_time_sec=start_time,
        end_time_sec=end_time,
        frames=frames if valid_frames > 0 else [],
        other_persons_present=False,
        average_evidence=evidence,
        majority_spatial_position=spatial,
    )


def test_initial_state(state_config):
    engine = StateEngine(state_config)
    assert engine.current_activity == ActivityState.UNKNOWN
    assert engine.current_bed_context == BedContext.UNSET


def test_state_confirmation_and_hysteresis(state_config):
    engine = StateEngine(state_config)

    # Send strong Lying evidence
    ev_lying = ActivityEvidence(lying=0.9, sitting=0.1)
    obs1 = create_mock_obs(0.0, 2.0, ev_lying, "INSIDE")
    act, ctx = engine.process_observation(obs1)

    assert act == ActivityState.LYING_IN_BED
    assert ctx == BedContext.IN_BED

    # Send weak sitting evidence (should not change due to hysteresis)
    # Lying score (1 obs) = 0.9 * 1.5 (hysteresis) = 1.35
    # Sitting score (1 obs) = 0.6
    ev_weak_sit = ActivityEvidence(lying=0.0, sitting=0.6)
    obs2 = create_mock_obs(2.0, 4.0, ev_weak_sit, "INSIDE")
    act, ctx = engine.process_observation(obs2)

    assert act == ActivityState.LYING_IN_BED

    # Send strong sitting evidence (should overcome hysteresis)
    # Window is now obs1, obs2, obs3 (6 seconds total, but config is 4s window)
    # Wait, the window logic pops obs1 because 6.0 - 0.0 > 4.0.
    ev_strong_sit = ActivityEvidence(lying=0.0, sitting=0.9)
    obs3 = create_mock_obs(4.0, 6.0, ev_strong_sit, "INSIDE")
    act, ctx = engine.process_observation(obs3)

    assert act == ActivityState.SITTING_ON_BED


def test_unknown_fallback(state_config):
    engine = StateEngine(state_config)

    # Valid observation
    ev = ActivityEvidence(lying=0.9)
    obs1 = create_mock_obs(0.0, 2.0, ev)
    engine.process_observation(obs1)
    assert engine.current_activity == ActivityState.LYING_IN_BED

    # No target frames -> falls back to UNKNOWN
    obs2 = create_mock_obs(2.0, 4.0, ActivityEvidence(), valid_frames=0)
    act, _ = engine.process_observation(obs2)
    assert act == ActivityState.UNKNOWN
