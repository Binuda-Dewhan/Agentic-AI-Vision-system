"""
Comprehensive tests for the hardening pass.

Covers:
  A. Video timing (segment duration, adjacent segments, final segment, duration sum)
  B. State logic (all states, bed context)
  C. Bed events (exit, false exit, return, repeated exits)
  D. Temporal reasoning (context building, VLM escalation conditions)
  E. Safety rules (all 6 rules)
  F. Reporting (duration sums, final state, bed summary)
"""

import pytest

from sleep_monitor.agent.temporal_context import StateHistoryEntry, TemporalContext
from sleep_monitor.config.settings import (
    BedEventsConfig,
    SafetyRulesConfig,
    TemporalStateConfig,
)
from sleep_monitor.engine.event_engine import BedEventEngine
from sleep_monitor.engine.safety_engine import SafetyEngine
from sleep_monitor.engine.state_engine import StateEngine
from sleep_monitor.engine.timeline_engine import TimelineEngine
from sleep_monitor.schemas.events import BedEvent
from sleep_monitor.schemas.perception import (
    ActivityEvidence,
    BBox,
    FrameObservation,
    Observation,
)
from sleep_monitor.schemas.state import ActivityState, BedContext, SafetyDecision


# ============================================================
# Helpers
# ============================================================

def make_obs(
    seg_id: int,
    start: float,
    end: float,
    evidence: ActivityEvidence,
    spatial: str = "UNKNOWN",
    valid_frames: int = 2,
) -> Observation:
    """Build a minimal Observation for testing."""
    frames = []
    for i in range(valid_frames):
        t = start + i * (end - start) / max(valid_frames, 1)
        frames.append(
            FrameObservation(
                frame_index=i,
                timestamp_sec=t,
                bbox=BBox(x1=0, y1=0, x2=100, y2=200) if valid_frames > 0 else None,
                evidence=evidence,
            )
        )
    return Observation(
        segment_id=seg_id,
        start_time_sec=start,
        end_time_sec=end,
        frames=frames,
        average_evidence=evidence,
        majority_spatial_position=spatial,
    )


# ============================================================
# A. VIDEO TIMING TESTS
# ============================================================

class TestVideoTiming:
    """Tests that segment timing produces contiguous, gap-free segments."""

    def test_segment_duration_2fps_2sec(self):
        """2 FPS × 2-sec segments → each segment should cover exactly 2.0s."""
        engine = TimelineEngine()
        # Simulate 3 contiguous segments: [0,2), [2,4), [4,6)
        engine.process_segment(0.0, 2.0, ActivityState.LYING_IN_BED, BedContext.IN_BED, None)
        engine.process_segment(2.0, 4.0, ActivityState.LYING_IN_BED, BedContext.IN_BED, None)
        engine.process_segment(4.0, 6.0, ActivityState.LYING_IN_BED, BedContext.IN_BED, None)

        report = engine.finalize()
        assert report.activity_durations[ActivityState.LYING_IN_BED.value] == 6.0

    def test_adjacent_segments_no_gap(self):
        """Adjacent segments must be contiguous — no gap between them."""
        engine = TimelineEngine()
        engine.process_segment(0.0, 2.0, ActivityState.LYING_IN_BED, BedContext.IN_BED, None)
        engine.process_segment(2.0, 4.0, ActivityState.STANDING, BedContext.IN_BED, None)

        report = engine.finalize()
        # Timeline segments should be [0, 2) and [2, 4)
        assert len(report.timeline) == 2
        assert report.timeline[0].end_time_sec == 2.0
        assert report.timeline[1].start_time_sec == 2.0

    def test_final_partial_segment(self):
        """Final segment may be shorter; duration should still be correct."""
        engine = TimelineEngine()
        engine.process_segment(0.0, 2.0, ActivityState.LYING_IN_BED, BedContext.IN_BED, None)
        engine.process_segment(2.0, 3.5, ActivityState.LYING_IN_BED, BedContext.IN_BED, None)  # partial

        report = engine.finalize()
        total = sum(report.activity_durations.values())
        assert total == pytest.approx(3.5)

    def test_duration_sum_matches_video(self):
        """Sum of all activity durations should equal total segment coverage."""
        engine = TimelineEngine()
        engine.process_segment(0.0, 2.0, ActivityState.LYING_IN_BED, BedContext.IN_BED, None)
        engine.process_segment(2.0, 4.0, ActivityState.SITTING_ON_BED, BedContext.IN_BED, None)
        engine.process_segment(4.0, 6.0, ActivityState.STANDING, BedContext.OUT_OF_BED, None)
        engine.process_segment(6.0, 8.0, ActivityState.WALKING, BedContext.OUT_OF_BED, None)

        report = engine.finalize()
        total = sum(report.activity_durations.values())
        assert total == pytest.approx(8.0)

    def test_out_of_bed_not_in_activity_durations(self):
        """OUT_OF_BED should NOT appear in activity_durations (tracked in bed_summary)."""
        engine = TimelineEngine()
        engine.process_segment(0.0, 2.0, ActivityState.WALKING, BedContext.OUT_OF_BED, None)

        report = engine.finalize()
        assert "OUT_OF_BED" not in report.activity_durations
        assert report.bed_summary.time_out_of_bed == 2.0


# ============================================================
# B. STATE LOGIC TESTS
# ============================================================

class TestStateLogic:
    """Tests for state engine with different evidence patterns."""

    @pytest.fixture
    def engine(self):
        return StateEngine(TemporalStateConfig(
            state_confirmation_window_sec=4,
            state_transition_hysteresis=1.5,
            confidence_threshold=0.3,
        ))

    def test_lying_in_bed(self, engine):
        obs = make_obs(0, 0.0, 2.0, ActivityEvidence(lying=0.9), spatial="INSIDE")
        act, ctx = engine.process_observation(obs)
        assert act == ActivityState.LYING_IN_BED

    def test_sitting_on_bed(self, engine):
        obs = make_obs(0, 0.0, 2.0, ActivityEvidence(sitting=0.9), spatial="INSIDE")
        act, ctx = engine.process_observation(obs)
        assert act == ActivityState.SITTING_ON_BED

    def test_sitting_outside_bed(self, engine):
        obs = make_obs(0, 0.0, 2.0, ActivityEvidence(sitting=0.9), spatial="OUTSIDE")
        act, ctx = engine.process_observation(obs)
        assert act == ActivityState.SITTING_OUTSIDE_BED

    def test_standing(self, engine):
        obs = make_obs(0, 0.0, 2.0, ActivityEvidence(standing=0.9), spatial="OUTSIDE")
        act, ctx = engine.process_observation(obs)
        assert act == ActivityState.STANDING

    def test_walking(self, engine):
        obs = make_obs(0, 0.0, 2.0, ActivityEvidence(walking=0.9), spatial="OUTSIDE")
        act, ctx = engine.process_observation(obs)
        assert act == ActivityState.WALKING

    def test_unknown_on_no_frames(self, engine):
        obs = make_obs(0, 0.0, 2.0, ActivityEvidence(), valid_frames=0)
        act, ctx = engine.process_observation(obs)
        assert act == ActivityState.UNKNOWN

    def test_initial_bed_context_in_bed(self, engine):
        obs = make_obs(0, 0.0, 2.0, ActivityEvidence(lying=0.9), spatial="INSIDE")
        _, ctx = engine.process_observation(obs)
        assert ctx == BedContext.IN_BED

    def test_initial_bed_context_out_of_bed(self, engine):
        obs = make_obs(0, 0.0, 2.0, ActivityEvidence(standing=0.9), spatial="OUTSIDE")
        _, ctx = engine.process_observation(obs)
        assert ctx == BedContext.OUT_OF_BED


# ============================================================
# C. BED EVENT TESTS
# ============================================================

class TestBedEvents:
    """Tests for bed exit and return detection with hysteresis."""

    @pytest.fixture
    def event_engine(self):
        return BedEventEngine(BedEventsConfig(
            bed_exit_hysteresis_sec=4,
            return_hysteresis_sec=2,
        ))

    def _walk_outside_obs(self, start, end):
        return make_obs(0, start, end, ActivityEvidence(walking=0.9), spatial="OUTSIDE")

    def _lying_inside_obs(self, start, end):
        return make_obs(0, start, end, ActivityEvidence(lying=0.9), spatial="INSIDE")

    def test_normal_bed_exit(self, event_engine):
        """Person walks outside bed for longer than hysteresis → BED_EXIT."""
        # First observation: walking outside, starts potential exit
        obs1 = self._walk_outside_obs(0.0, 2.0)
        ctx1, ev1 = event_engine.process(
            ActivityState.WALKING, BedContext.IN_BED, obs1, ActivityState.LYING_IN_BED
        )
        assert ctx1 == BedContext.IN_BED
        assert ev1 is None  # Not confirmed yet

        # Second observation: still outside, exceeds hysteresis (4s)
        obs2 = self._walk_outside_obs(2.0, 5.0)
        ctx2, ev2 = event_engine.process(
            ActivityState.WALKING, BedContext.IN_BED, obs2, ActivityState.WALKING
        )
        assert ctx2 == BedContext.OUT_OF_BED
        assert ev2 is not None
        assert ev2.event_type == "BED_EXIT"

    def test_false_exit_brief_standing(self, event_engine):
        """Person stands briefly then lies back → no BED_EXIT."""
        # Start potential exit
        obs1 = make_obs(0, 0.0, 2.0, ActivityEvidence(standing=0.8), spatial="OUTSIDE")
        ctx1, ev1 = event_engine.process(
            ActivityState.STANDING, BedContext.IN_BED, obs1
        )
        assert ev1 is None

        # Person lies back before hysteresis threshold
        obs2 = self._lying_inside_obs(2.0, 4.0)
        ctx2, ev2 = event_engine.process(
            ActivityState.LYING_IN_BED, BedContext.IN_BED, obs2
        )
        assert ctx2 == BedContext.IN_BED
        assert ev2 is None
        assert event_engine.potential_exit_start_time is None  # Reset

    def test_edge_sitting_no_exit(self, event_engine):
        """Person sits on bed edge → no BED_EXIT (spatial is ON_EDGE, not OUTSIDE)."""
        obs = make_obs(0, 0.0, 6.0, ActivityEvidence(sitting=0.9), spatial="ON_EDGE")
        ctx, ev = event_engine.process(
            ActivityState.SITTING_ON_BED, BedContext.IN_BED, obs
        )
        assert ev is None
        assert ctx == BedContext.IN_BED

    def test_return_to_bed(self, event_engine):
        """Person returns to bed → RETURN_TO_BED after hysteresis."""
        obs1 = self._lying_inside_obs(0.0, 1.0)
        ctx1, ev1 = event_engine.process(
            ActivityState.LYING_IN_BED, BedContext.OUT_OF_BED, obs1
        )
        assert ev1 is None  # Not confirmed yet

        obs2 = self._lying_inside_obs(1.0, 3.0)
        ctx2, ev2 = event_engine.process(
            ActivityState.LYING_IN_BED, BedContext.OUT_OF_BED, obs2
        )
        assert ctx2 == BedContext.IN_BED
        assert ev2 is not None
        assert ev2.event_type == "RETURN_TO_BED"

    def test_leave_and_return(self, event_engine):
        """Full cycle: exit → walk → return → confirmed."""
        # Exit
        obs1 = self._walk_outside_obs(0.0, 2.0)
        event_engine.process(ActivityState.WALKING, BedContext.IN_BED, obs1, ActivityState.LYING_IN_BED)
        obs2 = self._walk_outside_obs(2.0, 5.0)
        ctx, exit_ev = event_engine.process(ActivityState.WALKING, BedContext.IN_BED, obs2)
        assert exit_ev is not None and exit_ev.event_type == "BED_EXIT"

        # Return
        obs3 = self._lying_inside_obs(5.0, 6.0)
        event_engine.process(ActivityState.LYING_IN_BED, BedContext.OUT_OF_BED, obs3)
        obs4 = self._lying_inside_obs(6.0, 8.0)
        ctx, ret_ev = event_engine.process(ActivityState.LYING_IN_BED, BedContext.OUT_OF_BED, obs4)
        assert ret_ev is not None and ret_ev.event_type == "RETURN_TO_BED"
        assert ctx == BedContext.IN_BED

    def test_repeated_exits(self, event_engine):
        """Multiple exit-return cycles produce correct event counts."""
        events = []
        context = BedContext.IN_BED
        prev_act = ActivityState.LYING_IN_BED

        for cycle in range(3):
            base = cycle * 20.0
            # Exit
            obs1 = self._walk_outside_obs(base, base + 2.0)
            context, ev = event_engine.process(ActivityState.WALKING, context, obs1, prev_act)
            obs2 = self._walk_outside_obs(base + 2.0, base + 5.0)
            context, ev = event_engine.process(ActivityState.WALKING, context, obs2)
            if ev:
                events.append(ev)

            # Return
            obs3 = self._lying_inside_obs(base + 10.0, base + 11.0)
            context, ev = event_engine.process(ActivityState.LYING_IN_BED, context, obs3)
            obs4 = self._lying_inside_obs(base + 11.0, base + 13.0)
            context, ev = event_engine.process(ActivityState.LYING_IN_BED, context, obs4)
            if ev:
                events.append(ev)
            prev_act = ActivityState.LYING_IN_BED

        exits = [e for e in events if e.event_type == "BED_EXIT"]
        returns = [e for e in events if e.event_type == "RETURN_TO_BED"]
        assert len(exits) == 3
        assert len(returns) == 3


# ============================================================
# D. TEMPORAL REASONING TESTS
# ============================================================

class TestTemporalReasoning:
    """Tests for temporal context building and VLM escalation conditions."""

    def test_temporal_context_format(self):
        """TemporalContext.format_for_prompt() produces readable text."""
        ctx = TemporalContext(
            current_segment_id=5,
            current_start_sec=10.0,
            current_end_sec=12.0,
            current_evidence=ActivityEvidence(lying=0.3, sitting=0.3),
            current_spatial="ON_EDGE",
            previous_evidence=ActivityEvidence(lying=0.8),
            previous_spatial="INSIDE",
            previous_activity="LYING_IN_BED",
            next_evidence=ActivityEvidence(standing=0.7),
            next_spatial="OUTSIDE",
            bed_context="IN_BED",
            state_history=[
                StateHistoryEntry(time_sec=8.0, activity="LYING_IN_BED", bed_context="IN_BED"),
                StateHistoryEntry(time_sec=10.0, activity="UNKNOWN", bed_context="IN_BED"),
            ],
        )

        text = ctx.format_for_prompt()
        assert "Current Segment 5" in text
        assert "Previous Segment" in text
        assert "Following Segment" in text
        assert "State History" in text
        assert "LYING_IN_BED" in text

    def test_temporal_context_no_next(self):
        """When next segment is unavailable, it is simply omitted."""
        ctx = TemporalContext(
            current_segment_id=0,
            current_start_sec=0.0,
            current_end_sec=2.0,
            current_evidence=ActivityEvidence(lying=0.9),
        )
        text = ctx.format_for_prompt()
        assert "Following Segment" not in text

    def test_temporal_context_pending_events(self):
        """Pending event candidates are included in the prompt."""
        ctx = TemporalContext(
            current_segment_id=0,
            current_start_sec=0.0,
            current_end_sec=2.0,
            current_evidence=ActivityEvidence(),
            pending_exit_since_sec=1.0,
        )
        text = ctx.format_for_prompt()
        assert "Pending BED_EXIT" in text


# ============================================================
# E. SAFETY RULE TESTS
# ============================================================

class TestSafetyRules:
    """Tests for all 6 safety rules."""

    @pytest.fixture
    def safety_config(self):
        return SafetyRulesConfig(
            monitor_confidence_threshold=0.6,
            monitor_bed_edge_duration_sec=120,
            out_of_bed_monitor_sec=300,
            alert_out_of_bed_duration_sec=900,
            alert_unknown_duration_sec=300,
            max_exits_before_alert=3,
        )

    def _timeline_with_durations(self, **durations):
        """Create a TimelineEngine with preset activity durations."""
        tl = TimelineEngine()
        for state, dur in durations.items():
            tl.activity_durations[state] = dur
        return tl

    def test_rule1_unknown_monitor(self, safety_config):
        """Current UNKNOWN → MONITOR."""
        engine = SafetyEngine(safety_config)
        tl = self._timeline_with_durations(UNKNOWN=10.0)
        decision = engine.evaluate(ActivityState.UNKNOWN, BedContext.IN_BED, tl)
        assert decision == SafetyDecision.MONITOR

    def test_rule2_prolonged_bed_edge_monitor(self, safety_config):
        """Prolonged SITTING_ON_BED → MONITOR."""
        engine = SafetyEngine(safety_config)
        tl = self._timeline_with_durations(SITTING_ON_BED=150.0)
        decision = engine.evaluate(ActivityState.SITTING_ON_BED, BedContext.IN_BED, tl)
        assert decision == SafetyDecision.MONITOR

    def test_rule3_prolonged_out_of_bed_alert(self, safety_config):
        """Prolonged out-of-bed → ALERT."""
        engine = SafetyEngine(safety_config)
        tl = self._timeline_with_durations(WALKING=1000.0)
        tl.current_out_duration = 1000.0
        decision = engine.evaluate(ActivityState.WALKING, BedContext.OUT_OF_BED, tl)
        assert decision == SafetyDecision.ALERT

    def test_rule3_moderate_out_of_bed_monitor(self, safety_config):
        """Moderate out-of-bed → MONITOR (before ALERT threshold)."""
        engine = SafetyEngine(safety_config)
        tl = self._timeline_with_durations(WALKING=400.0)
        tl.current_out_duration = 400.0
        decision = engine.evaluate(ActivityState.WALKING, BedContext.OUT_OF_BED, tl)
        assert decision == SafetyDecision.MONITOR

    def test_rule4_extended_unknown_alert(self, safety_config):
        """Extended UNKNOWN duration → ALERT."""
        engine = SafetyEngine(safety_config)
        tl = self._timeline_with_durations(UNKNOWN=350.0)
        decision = engine.evaluate(ActivityState.LYING_IN_BED, BedContext.IN_BED, tl)
        assert decision == SafetyDecision.ALERT

    def test_rule5_fall_detection_alert(self, safety_config):
        """Lying + OUT_OF_BED context → ALERT (fall-like)."""
        engine = SafetyEngine(safety_config)
        tl = self._timeline_with_durations()
        decision = engine.evaluate(
            ActivityState.LYING_IN_BED, BedContext.OUT_OF_BED, tl
        )
        assert decision == SafetyDecision.ALERT

    def test_rule6_multiple_exits_alert(self, safety_config):
        """More than max_exits → ALERT."""
        engine = SafetyEngine(safety_config)
        tl = self._timeline_with_durations()
        for _ in range(4):
            tl.bed_events.append(BedEvent(event_type="BED_EXIT", timestamp_sec=0, confirmed_at_sec=0))
        decision = engine.evaluate(ActivityState.LYING_IN_BED, BedContext.IN_BED, tl)
        assert decision == SafetyDecision.ALERT

    def test_normal_recovery(self, safety_config):
        """Person returns to bed → recovers to NORMAL."""
        engine = SafetyEngine(safety_config)
        tl = self._timeline_with_durations(LYING_IN_BED=100.0)
        decision = engine.evaluate(ActivityState.LYING_IN_BED, BedContext.IN_BED, tl)
        assert decision == SafetyDecision.NORMAL


# ============================================================
# F. REPORTING TESTS
# ============================================================

class TestReporting:
    """Tests for the final report structure and correctness."""

    def test_report_duration_sum(self):
        """Activity durations must sum to total segment coverage."""
        engine = TimelineEngine()
        engine.process_segment(0.0, 5.0, ActivityState.LYING_IN_BED, BedContext.IN_BED, None)
        engine.process_segment(5.0, 8.0, ActivityState.SITTING_ON_BED, BedContext.IN_BED, None)
        engine.process_segment(8.0, 12.0, ActivityState.WALKING, BedContext.OUT_OF_BED, None)
        engine.process_segment(12.0, 15.0, ActivityState.UNKNOWN, BedContext.OUT_OF_BED, None)

        report = engine.finalize()
        total = sum(report.activity_durations.values())
        assert total == pytest.approx(15.0)

    def test_report_final_state(self):
        """Final state matches the last timeline segment."""
        engine = TimelineEngine()
        engine.process_segment(0.0, 5.0, ActivityState.LYING_IN_BED, BedContext.IN_BED, None)
        engine.process_segment(5.0, 8.0, ActivityState.WALKING, BedContext.OUT_OF_BED, None)
        report = engine.finalize()
        assert report.target_person.final_state == "WALKING"

    def test_report_bed_summary(self):
        """Bed summary correctly tracks in/out time and event counts."""
        engine = TimelineEngine()

        engine.process_segment(0.0, 5.0, ActivityState.LYING_IN_BED, BedContext.IN_BED, None)

        exit_event = BedEvent(event_type="BED_EXIT", timestamp_sec=5.0, confirmed_at_sec=7.0)
        engine.process_segment(5.0, 10.0, ActivityState.WALKING, BedContext.OUT_OF_BED, exit_event)

        return_event = BedEvent(event_type="RETURN_TO_BED", timestamp_sec=10.0, confirmed_at_sec=12.0)
        engine.process_segment(10.0, 15.0, ActivityState.LYING_IN_BED, BedContext.IN_BED, return_event)

        report = engine.finalize()

        assert report.bed_summary.time_in_bed == 10.0
        assert report.bed_summary.time_out_of_bed == 5.0
        assert report.bed_summary.exit_count == 1
        assert report.bed_summary.return_count == 1
        assert report.bed_summary.longest_out_of_bed == 5.0

    def test_report_event_counts(self):
        """Event counts match the number of recorded events."""
        engine = TimelineEngine()
        engine.process_segment(0.0, 2.0, ActivityState.WALKING, BedContext.OUT_OF_BED,
                               BedEvent(event_type="BED_EXIT", timestamp_sec=0, confirmed_at_sec=2))
        engine.process_segment(2.0, 4.0, ActivityState.LYING_IN_BED, BedContext.IN_BED,
                               BedEvent(event_type="RETURN_TO_BED", timestamp_sec=2, confirmed_at_sec=4))
        report = engine.finalize()
        assert report.bed_summary.exit_count == 1
        assert report.bed_summary.return_count == 1

    def test_timeline_merges_contiguous_states(self):
        """Contiguous segments with the same state are merged into one timeline entry."""
        engine = TimelineEngine()
        engine.process_segment(0.0, 2.0, ActivityState.LYING_IN_BED, BedContext.IN_BED, None)
        engine.process_segment(2.0, 4.0, ActivityState.LYING_IN_BED, BedContext.IN_BED, None)
        engine.process_segment(4.0, 6.0, ActivityState.LYING_IN_BED, BedContext.IN_BED, None)

        report = engine.finalize()
        assert len(report.timeline) == 1
        assert report.timeline[0].start_time_sec == 0.0
        assert report.timeline[0].end_time_sec == 6.0
