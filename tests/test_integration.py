"""
Integration tests for the hardening pass.

These tests exercise the ACTUAL components end-to-end rather than
testing isolated helpers. They verify:

  A. VideoIngester produces correct segment timestamps
  B. Pipeline._should_investigate() triggers on all 3 uncertainty conditions
  C. Pipeline._build_temporal_context() wires previous/current/next correctly
  D. SafetyEngine evaluates all 6 rules through the actual evaluate() method
  E. VideoIngester segment timing is gap-free and sums to video duration
"""

import tempfile
from collections import deque
from pathlib import Path
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest

from sleep_monitor.agent.temporal_context import StateHistoryEntry, TemporalContext
from sleep_monitor.config.settings import (
    SafetyRulesConfig,
    Settings,
    TemporalStateConfig,
)
from sleep_monitor.engine.safety_engine import SafetyEngine
from sleep_monitor.engine.timeline_engine import TimelineEngine
from sleep_monitor.pipeline import Pipeline
from sleep_monitor.schemas.events import BedEvent
from sleep_monitor.schemas.perception import ActivityEvidence, BBox, FrameObservation, Observation
from sleep_monitor.schemas.state import ActivityState, BedContext, SafetyDecision
from sleep_monitor.video.ingestion import VideoIngester


# ============================================================
# Helpers
# ============================================================

def make_test_video(path: str, fps: float = 10.0, duration_sec: float = 6.0, width: int = 64, height: int = 64):
    """Create a minimal test video file for VideoIngester tests."""
    total_frames = int(fps * duration_sec)
    fourcc = cv2.VideoWriter.fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, fps, (width, height))
    for i in range(total_frames):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[:, :, 0] = i % 256  # vary blue channel
        writer.write(frame)
    writer.release()


def make_obs(seg_id, start, end, evidence, spatial="UNKNOWN", valid_frames=2):
    """Build a minimal Observation for testing."""
    frames = []
    for i in range(valid_frames):
        t = start + i * (end - start) / max(valid_frames, 1)
        frames.append(
            FrameObservation(
                frame_index=i,
                timestamp_sec=t,
                bbox=BBox(x1=0, y1=0, x2=100, y2=200),
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
# A. VideoIngester segment timestamp tests
# ============================================================

class TestVideoIngesterTimestamps:
    """Verify VideoIngester.generate_segments() produces correct timestamps."""

    @pytest.fixture
    def video_path(self, tmp_path):
        path = str(tmp_path / "test.mp4")
        make_test_video(path, fps=10.0, duration_sec=6.0)
        return path

    def test_segments_are_contiguous(self, video_path):
        """Adjacent segments must have no gaps: seg[i].end == seg[i+1].start."""
        ingester = VideoIngester(video_path)
        segments = list(ingester.generate_segments(target_fps=2.0, segment_duration_sec=2.0))

        assert len(segments) >= 2, f"Expected >=2 segments, got {len(segments)}"
        for i in range(len(segments) - 1):
            assert segments[i].end_time_sec == pytest.approx(
                segments[i + 1].start_time_sec, abs=0.01
            ), f"Gap between segment {i} end ({segments[i].end_time_sec}) and segment {i+1} start ({segments[i+1].start_time_sec})"

    def test_segment_duration_matches_config(self, video_path):
        """Each full segment should cover exactly segment_duration_sec."""
        ingester = VideoIngester(video_path)
        segments = list(ingester.generate_segments(target_fps=2.0, segment_duration_sec=2.0))

        # All segments except possibly the last should be 2.0s
        for seg in segments[:-1]:
            duration = seg.end_time_sec - seg.start_time_sec
            assert duration == pytest.approx(2.0, abs=0.01), (
                f"Segment {seg.segment_id} duration is {duration}, expected 2.0"
            )

    def test_total_coverage_equals_video_duration(self, video_path):
        """Sum of all segment durations should equal video duration."""
        ingester = VideoIngester(video_path)
        metadata = ingester.get_metadata()
        segments = list(ingester.generate_segments(target_fps=2.0, segment_duration_sec=2.0))

        total_coverage = sum(s.end_time_sec - s.start_time_sec for s in segments)
        assert total_coverage == pytest.approx(metadata.duration_sec, abs=0.1), (
            f"Total coverage {total_coverage} != video duration {metadata.duration_sec}"
        )

    def test_first_segment_starts_at_zero(self, video_path):
        """First segment must start at t=0."""
        ingester = VideoIngester(video_path)
        segments = list(ingester.generate_segments(target_fps=2.0, segment_duration_sec=2.0))

        assert segments[0].start_time_sec == pytest.approx(0.0, abs=0.01)

    def test_last_segment_ends_at_video_duration(self, video_path):
        """Last segment must end at or near the video duration."""
        ingester = VideoIngester(video_path)
        metadata = ingester.get_metadata()
        segments = list(ingester.generate_segments(target_fps=2.0, segment_duration_sec=2.0))

        assert segments[-1].end_time_sec == pytest.approx(metadata.duration_sec, abs=0.1), (
            f"Last segment ends at {segments[-1].end_time_sec}, expected {metadata.duration_sec}"
        )

    def test_end_time_is_not_last_frame_timestamp(self, video_path):
        """end_time must NOT be the timestamp of the last frame in the segment."""
        ingester = VideoIngester(video_path)
        segments = list(ingester.generate_segments(target_fps=2.0, segment_duration_sec=2.0))

        for seg in segments[:-1]:
            last_frame_ts = seg.frames[-1].timestamp_sec
            # end_time should be start + segment_duration, not last_frame.timestamp
            assert seg.end_time_sec != pytest.approx(last_frame_ts, abs=0.001) or \
                   seg.end_time_sec == pytest.approx(seg.start_time_sec + 2.0, abs=0.01), (
                f"Segment {seg.segment_id}: end_time appears to be last frame timestamp"
            )


# ============================================================
# B. Pipeline._should_investigate() triggers
# ============================================================

class TestPipelineUncertaintyTriggers:
    """Verify that _should_investigate triggers on all 3 documented conditions."""

    @pytest.fixture
    def pipeline(self):
        """Create a Pipeline with mocked heavy components."""
        with patch("sleep_monitor.pipeline.PersonDetector"), \
             patch("sleep_monitor.pipeline.PoseEstimator"), \
             patch("sleep_monitor.pipeline.SpatialAnalyzer"), \
             patch("sleep_monitor.pipeline.InvestigationAgent"):
            p = Pipeline(Settings())
        return p

    def test_trigger_on_unknown(self, pipeline):
        """UNKNOWN state → should investigate."""
        obs = make_obs(0, 0.0, 2.0, ActivityEvidence())
        assert pipeline._should_investigate(ActivityState.UNKNOWN, obs) is True

    def test_trigger_on_ambiguous_scores(self, pipeline):
        """Two competing scores within 80% → should investigate."""
        # lying=0.5, sitting=0.45 → ratio=0.9 > 0.8
        obs = make_obs(0, 0.0, 2.0, ActivityEvidence(lying=0.5, sitting=0.45))
        assert pipeline._should_investigate(ActivityState.LYING_IN_BED, obs) is True

    def test_no_trigger_on_clear_evidence(self, pipeline):
        """One dominant score → should NOT investigate."""
        obs = make_obs(0, 0.0, 2.0, ActivityEvidence(lying=0.9, sitting=0.1))
        assert pipeline._should_investigate(ActivityState.LYING_IN_BED, obs) is False

    def test_trigger_on_weak_state_transition(self, pipeline):
        """State changed with weak evidence score → should investigate."""
        # Seed state history with a previous state
        pipeline._state_history.append(
            StateHistoryEntry(time_sec=0.0, activity="LYING_IN_BED", bed_context="IN_BED")
        )
        # New state is STANDING but evidence is weak (0.4 < 0.5)
        obs = make_obs(0, 2.0, 4.0, ActivityEvidence(standing=0.4))
        assert pipeline._should_investigate(ActivityState.STANDING, obs) is True

    def test_no_trigger_on_strong_transition(self, pipeline):
        """State changed with strong evidence → should NOT investigate."""
        pipeline._state_history.append(
            StateHistoryEntry(time_sec=0.0, activity="LYING_IN_BED", bed_context="IN_BED")
        )
        obs = make_obs(0, 2.0, 4.0, ActivityEvidence(standing=0.9))
        assert pipeline._should_investigate(ActivityState.STANDING, obs) is False


# ============================================================
# C. Pipeline._build_temporal_context() wiring
# ============================================================

class TestPipelineTemporalContextWiring:
    """Verify that _build_temporal_context correctly wires all fields."""

    @pytest.fixture
    def pipeline(self):
        with patch("sleep_monitor.pipeline.PersonDetector"), \
             patch("sleep_monitor.pipeline.PoseEstimator"), \
             patch("sleep_monitor.pipeline.SpatialAnalyzer"), \
             patch("sleep_monitor.pipeline.InvestigationAgent"):
            p = Pipeline(Settings())
        return p

    def test_current_segment_fields(self, pipeline):
        """Current segment evidence and spatial are correctly set."""
        obs = make_obs(3, 6.0, 8.0, ActivityEvidence(lying=0.7), spatial="INSIDE")
        ctx = pipeline._build_temporal_context(obs, ActivityState.LYING_IN_BED)

        assert ctx.current_segment_id == 3
        assert ctx.current_start_sec == 6.0
        assert ctx.current_end_sec == 8.0
        assert ctx.current_evidence.lying == 0.7
        assert ctx.current_spatial == "INSIDE"

    def test_previous_segment_fields(self, pipeline):
        """Previous segment data is passed through when available."""
        prev_obs = make_obs(2, 4.0, 6.0, ActivityEvidence(sitting=0.8), spatial="ON_EDGE")
        curr_obs = make_obs(3, 6.0, 8.0, ActivityEvidence(lying=0.7), spatial="INSIDE")

        ctx = pipeline._build_temporal_context(
            curr_obs, ActivityState.LYING_IN_BED,
            prev_observation=prev_obs, prev_activity=ActivityState.SITTING_ON_BED,
        )

        assert ctx.previous_evidence is not None
        assert ctx.previous_evidence.sitting == 0.8
        assert ctx.previous_spatial == "ON_EDGE"
        assert ctx.previous_activity == "SITTING_ON_BED"

    def test_next_segment_fields(self, pipeline):
        """Next/following segment is included when buffered."""
        curr_obs = make_obs(3, 6.0, 8.0, ActivityEvidence(lying=0.7), spatial="INSIDE")
        next_obs = make_obs(4, 8.0, 10.0, ActivityEvidence(standing=0.6), spatial="OUTSIDE")

        ctx = pipeline._build_temporal_context(
            curr_obs, ActivityState.LYING_IN_BED,
            next_observation=next_obs,
        )

        assert ctx.next_evidence is not None
        assert ctx.next_evidence.standing == 0.6
        assert ctx.next_spatial == "OUTSIDE"

    def test_no_next_when_not_buffered(self, pipeline):
        """Next fields are None when no look-ahead is available."""
        curr_obs = make_obs(3, 6.0, 8.0, ActivityEvidence(lying=0.7))
        ctx = pipeline._build_temporal_context(curr_obs, ActivityState.LYING_IN_BED)

        assert ctx.next_evidence is None
        assert ctx.next_spatial is None

    def test_state_history_included(self, pipeline):
        """Recent state history entries are included in the context."""
        pipeline._state_history.append(
            StateHistoryEntry(time_sec=2.0, activity="LYING_IN_BED", bed_context="IN_BED")
        )
        pipeline._state_history.append(
            StateHistoryEntry(time_sec=4.0, activity="SITTING_ON_BED", bed_context="IN_BED")
        )

        curr_obs = make_obs(3, 6.0, 8.0, ActivityEvidence(lying=0.7))
        ctx = pipeline._build_temporal_context(curr_obs, ActivityState.LYING_IN_BED)

        assert len(ctx.state_history) == 2
        assert ctx.state_history[0].activity == "LYING_IN_BED"
        assert ctx.state_history[1].activity == "SITTING_ON_BED"

    def test_bed_context_included(self, pipeline):
        """Current bed context is populated from the state engine."""
        curr_obs = make_obs(3, 6.0, 8.0, ActivityEvidence(lying=0.7))
        ctx = pipeline._build_temporal_context(curr_obs, ActivityState.LYING_IN_BED)
        # Default bed context from state engine should be present
        assert ctx.bed_context is not None

    def test_formatted_prompt_contains_all_sections(self, pipeline):
        """The formatted prompt string includes all temporal sections."""
        prev_obs = make_obs(2, 4.0, 6.0, ActivityEvidence(sitting=0.8), spatial="ON_EDGE")
        curr_obs = make_obs(3, 6.0, 8.0, ActivityEvidence(lying=0.4, sitting=0.35), spatial="INSIDE")
        next_obs = make_obs(4, 8.0, 10.0, ActivityEvidence(standing=0.6), spatial="OUTSIDE")

        pipeline._state_history.append(
            StateHistoryEntry(time_sec=4.0, activity="SITTING_ON_BED", bed_context="IN_BED")
        )

        ctx = pipeline._build_temporal_context(
            curr_obs, ActivityState.LYING_IN_BED,
            prev_observation=prev_obs, prev_activity=ActivityState.SITTING_ON_BED,
            next_observation=next_obs,
        )

        prompt = ctx.format_for_prompt()
        assert "Current Segment 3" in prompt
        assert "Previous Segment" in prompt
        assert "Following Segment" in prompt
        assert "State History" in prompt


# ============================================================
# D. SafetyEngine integration tests — all 6 rules
# ============================================================

class TestSafetyEngineIntegration:
    """Tests that exercise the full SafetyEngine.evaluate() pathway."""

    @pytest.fixture
    def config(self):
        return SafetyRulesConfig(
            monitor_confidence_threshold=0.6,
            monitor_bed_edge_duration_sec=120,
            out_of_bed_monitor_sec=300,
            alert_out_of_bed_duration_sec=900,
            alert_unknown_duration_sec=300,
            max_exits_before_alert=3,
        )

    def _make_timeline(self, **kwargs):
        """Create a TimelineEngine with custom durations and attributes."""
        tl = TimelineEngine()
        for k, v in kwargs.items():
            if k == "current_out_duration":
                tl.current_out_duration = v
            elif k == "bed_events":
                tl.bed_events = v
            else:
                tl.activity_durations[k] = v
        return tl

    def test_rule1_unknown_state_monitor(self, config):
        engine = SafetyEngine(config)
        tl = self._make_timeline(UNKNOWN=10.0)
        assert engine.evaluate(ActivityState.UNKNOWN, BedContext.IN_BED, tl) == SafetyDecision.MONITOR

    def test_rule2_prolonged_bed_edge_monitor(self, config):
        engine = SafetyEngine(config)
        tl = self._make_timeline(SITTING_ON_BED=150.0)
        assert engine.evaluate(ActivityState.SITTING_ON_BED, BedContext.IN_BED, tl) == SafetyDecision.MONITOR

    def test_rule3_prolonged_out_of_bed_alert(self, config):
        engine = SafetyEngine(config)
        tl = self._make_timeline(WALKING=1000.0, current_out_duration=1000.0)
        assert engine.evaluate(ActivityState.WALKING, BedContext.OUT_OF_BED, tl) == SafetyDecision.ALERT

    def test_rule3_moderate_out_of_bed_monitor(self, config):
        engine = SafetyEngine(config)
        tl = self._make_timeline(WALKING=400.0, current_out_duration=400.0)
        assert engine.evaluate(ActivityState.WALKING, BedContext.OUT_OF_BED, tl) == SafetyDecision.MONITOR

    def test_rule4_extended_unknown_alert(self, config):
        engine = SafetyEngine(config)
        tl = self._make_timeline(UNKNOWN=350.0)
        assert engine.evaluate(ActivityState.LYING_IN_BED, BedContext.IN_BED, tl) == SafetyDecision.ALERT

    def test_rule5_fall_detection_alert(self, config):
        engine = SafetyEngine(config)
        tl = self._make_timeline()
        assert engine.evaluate(ActivityState.LYING_IN_BED, BedContext.OUT_OF_BED, tl) == SafetyDecision.ALERT

    def test_rule6_multiple_exits_alert(self, config):
        engine = SafetyEngine(config)
        events = [BedEvent(event_type="BED_EXIT", timestamp_sec=i * 10, confirmed_at_sec=i * 10 + 5) for i in range(4)]
        tl = self._make_timeline(bed_events=events)
        assert engine.evaluate(ActivityState.LYING_IN_BED, BedContext.IN_BED, tl) == SafetyDecision.ALERT

    def test_normal_when_in_bed_no_issues(self, config):
        engine = SafetyEngine(config)
        tl = self._make_timeline(LYING_IN_BED=100.0)
        assert engine.evaluate(ActivityState.LYING_IN_BED, BedContext.IN_BED, tl) == SafetyDecision.NORMAL

    def test_alert_priority_over_monitor(self, config):
        """ALERT rules take priority over MONITOR conditions."""
        engine = SafetyEngine(config)
        # Fall-like state (ALERT) + would also trigger UNKNOWN MONITOR
        tl = self._make_timeline(UNKNOWN=10.0)
        result = engine.evaluate(ActivityState.LYING_IN_BED, BedContext.OUT_OF_BED, tl)
        assert result == SafetyDecision.ALERT

    def test_recovery_from_alert_to_normal(self, config):
        """After an ALERT, returning to bed should recover to NORMAL."""
        engine = SafetyEngine(config)
        # First: trigger ALERT
        tl = self._make_timeline()
        engine.evaluate(ActivityState.LYING_IN_BED, BedContext.OUT_OF_BED, tl)
        assert engine.current_decision == SafetyDecision.ALERT

        # Then: recover to NORMAL
        tl2 = self._make_timeline(LYING_IN_BED=100.0)
        result = engine.evaluate(ActivityState.LYING_IN_BED, BedContext.IN_BED, tl2)
        assert result == SafetyDecision.NORMAL


# ============================================================
# E. Pipeline -> Agent Integration
# ============================================================

class TestPipelineAgentIntegration:
    """Verify Pipeline actually passes TemporalContext to agent.investigate()."""

    @pytest.fixture
    def pipeline(self):
        with patch("sleep_monitor.pipeline.PersonDetector"), \
             patch("sleep_monitor.pipeline.PoseEstimator"), \
             patch("sleep_monitor.pipeline.SpatialAnalyzer"):
            p = Pipeline(Settings())
            # Mock the investigation agent
            p.agent = MagicMock()
            
            # Setup agent mock to return a known decision
            from sleep_monitor.agent.schemas import AgentDecision
            mock_decision = AgentDecision(
                confirmed_state=ActivityState.LYING_IN_BED,
                confidence=0.9,
                decision_type="VLM_CONFIRMED",
                reasoning="Looks like lying in bed.",
                vlm_used=True,
            )
            p.agent.investigate.return_value = mock_decision
        return p

    def test_pipeline_passes_temporal_context_to_agent(self, pipeline):
        """Pipeline must pass TemporalContext to the agent when investigating."""
        # 1. Setup observation that WILL trigger investigation (e.g. UNKNOWN state)
        obs = make_obs(1, 0.0, 2.0, ActivityEvidence())
        
        # 2. Process segment
        pipeline._process_segment_through_engines(
            observation=obs,
            segment_frames=[np.zeros((64, 64, 3), dtype=np.uint8)],
            previous_activity=ActivityState.UNKNOWN,
            prev_observation=None,
            next_observation=None,
        )
        
        # 3. Verify agent was called
        assert pipeline.agent.investigate.called, "Agent should have been called."
        
        # 4. Verify temporal_context was passed
        call_kwargs = pipeline.agent.investigate.call_args.kwargs
        assert "temporal_context" in call_kwargs, "TemporalContext missing from kwargs."
        
        temporal_context = call_kwargs["temporal_context"]
        assert isinstance(temporal_context, TemporalContext)
        assert temporal_context.current_segment_id == 1

