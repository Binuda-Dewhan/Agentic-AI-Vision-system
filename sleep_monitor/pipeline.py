import json
import logging
from collections import deque
from pathlib import Path

from sleep_monitor.agent.investigator import InvestigationAgent
from sleep_monitor.agent.temporal_context import (
    StateHistoryEntry,
    TemporalContext,
)
from sleep_monitor.config.settings import Settings
from sleep_monitor.engine.event_engine import BedEventEngine
from sleep_monitor.engine.safety_engine import SafetyEngine
from sleep_monitor.engine.state_engine import StateEngine
from sleep_monitor.engine.timeline_engine import TimelineEngine
from sleep_monitor.perception.detector import PersonDetector
from sleep_monitor.perception.observation_builder import ObservationBuilder
from sleep_monitor.perception.pose import PoseEstimator
from sleep_monitor.perception.spatial import SpatialAnalyzer
from sleep_monitor.perception.target_selector import TargetSelector
from sleep_monitor.schemas.perception import FrameObservation, Observation
from sleep_monitor.schemas.state import ActivityState
from sleep_monitor.video.ingestion import VideoIngester

logger = logging.getLogger(__name__)

# Maximum number of state history entries to keep for temporal context
_MAX_STATE_HISTORY = 10


class Pipeline:
    """
    End-to-end orchestrator that wires all engines together.

    Video → Perception → Observation → StateEngine → InvestigationAgent
         → BedEventEngine → TimelineEngine → SafetyEngine → FinalReport

    Temporal reasoning strategy:
      Processing is sequential. The pipeline uses a 1-segment buffer so that
      when the agent is invoked for segment N, it has access to:
        - previous segment (N-1) evidence and determined state
        - current segment (N) evidence
        - next segment (N+1) evidence (buffered look-ahead)
        - recent state history
        - bed context and pending event candidates

      The agent is only invoked on the BUFFERED segment once the next segment
      arrives, providing genuine future context without fabrication.
    """

    def __init__(self, config: Settings):
        self.config = config

        # Perception components
        self.detector = PersonDetector(
            model_path=config.perception.detection_model,
            confidence_threshold=config.perception.detection_confidence_threshold,
        )
        self.target_selector = TargetSelector(
            mode=config.target_person.mode,
            reassociation_distance_px=config.target_person.reassociation_distance_px,
        )
        self.pose_estimator = PoseEstimator(model_path=config.perception.pose_model)
        self.spatial_analyzer = SpatialAnalyzer(
            bed_polygon_points=config.bed_region.bed_region_polygon,
            edge_margin=config.bed_region.bed_edge_margin,
        )
        self.observation_builder = ObservationBuilder()

        # Engine components
        self.state_engine = StateEngine(config.temporal_state)
        self.event_engine = BedEventEngine(config.bed_events)
        self.timeline_engine = TimelineEngine()
        self.safety_engine = SafetyEngine(config.safety_rules)

        # Agent (optional)
        self.agent = InvestigationAgent(config.vlm)

        # State history for temporal context
        self._state_history: deque[StateHistoryEntry] = deque(
            maxlen=_MAX_STATE_HISTORY
        )

        # 1-segment look-ahead buffer for temporal agent reasoning
        self._buffered_segment: dict | None = None

    def _build_observation(self, segment, segment_frames_raw, visualize=False, previous_activity=None):
        """Run perception on a segment and return (observation, raw_frames)."""
        frame_observations = []
        any_other_persons = False

        for frame in segment.frames:
            raw_frame = frame.image
            if raw_frame is None:
                continue

            # Person detection + tracking
            persons = self.detector.detect_and_track(raw_frame)
            perception = self.target_selector.select_target(
                frame.frame_index, persons
            )

            if perception.other_persons_present:
                any_other_persons = True

            # Find primary target
            target = next(
                (p for p in perception.persons if p.is_primary_target), None
            )

            if target is None:
                frame_observations.append(
                    FrameObservation(
                        frame_index=frame.frame_index,
                        timestamp_sec=frame.timestamp_sec,
                    )
                )
                continue

            # Pose estimation on target
            pose = self.pose_estimator.estimate(raw_frame, target.bbox)

            # Spatial analysis
            spatial = self.spatial_analyzer.analyze(target.bbox)

            frame_observations.append(
                FrameObservation(
                    frame_index=frame.frame_index,
                    timestamp_sec=frame.timestamp_sec,
                    bbox=target.bbox,
                    pose=pose,
                    spatial=spatial,
                )
            )

            if visualize:
                import cv2
                import numpy as np
                annotated = self.detector.draw_annotations(raw_frame.copy(), persons)
                if self.config.bed_region.bed_region_polygon:
                    pts = np.array(self.config.bed_region.bed_region_polygon, np.int32)
                    pts = pts.reshape((-1, 1, 2))
                    cv2.polylines(annotated, [pts], True, (255, 0, 0), 2)
                
                state_text = f"Prev State: {previous_activity.value if previous_activity else 'UNKNOWN'}"
                cv2.putText(annotated, state_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # Manually resize the image to fit a reasonable screen size while preserving aspect ratio
                h, w = annotated.shape[:2]
                max_height = 720
                if h > max_height:
                    scale = max_height / h
                    new_w = int(w * scale)
                    new_h = int(h * scale)
                    annotated = cv2.resize(annotated, (new_w, new_h))
                    
                cv2.imshow("Analysis", annotated)
                cv2.waitKey(1)

        observation = self.observation_builder.build_observation(
            segment, frame_observations, any_other_persons
        )
        return observation

    def _should_investigate(
        self, current_activity: ActivityState, observation: Observation
    ) -> bool:
        """
        Determine whether the investigation agent should be invoked.

        Triggers (uncertainty-driven):
          1. Current state is UNKNOWN
          2. Top-two evidence scores are close (ambiguous classification)
          3. State just changed and the new score is low
          4. Evidence conflicts with recent state history
        """
        # Trigger 1: UNKNOWN state
        if current_activity == ActivityState.UNKNOWN:
            return True

        # Trigger 2: Ambiguous — top two scores are close
        ev = observation.average_evidence
        scores = {
            "lying": ev.lying,
            "sitting": ev.sitting,
            "standing": ev.standing,
            "walking": ev.walking,
        }
        sorted_scores = sorted(scores.values(), reverse=True)
        if len(sorted_scores) >= 2 and sorted_scores[0] > 0:
            ratio = sorted_scores[1] / sorted_scores[0]
            if ratio > 0.8:  # Second-best is within 80% of best
                return True

        # Trigger 3: State just changed and history shows the previous state
        # was held for a while (potential false transition)
        if (
            self._state_history
            and self._state_history[-1].activity != current_activity.value
        ):
            # Check if the evidence for the new state is weak
            state_to_score = {
                ActivityState.LYING_IN_BED.value: ev.lying,
                ActivityState.SITTING_ON_BED.value: ev.sitting,
                ActivityState.SITTING_OUTSIDE_BED.value: ev.sitting,
                ActivityState.STANDING.value: ev.standing,
                ActivityState.WALKING.value: ev.walking,
            }
            new_score = state_to_score.get(current_activity.value, 0.0)
            if new_score < 0.5:
                return True

        return False

    def _build_temporal_context(
        self,
        observation: Observation,
        current_activity: ActivityState,
        prev_observation: Observation | None = None,
        prev_activity: ActivityState | None = None,
        next_observation: Observation | None = None,
    ) -> TemporalContext:
        """Build a TemporalContext for the investigation agent."""
        ctx = TemporalContext(
            current_segment_id=observation.segment_id,
            current_start_sec=observation.start_time_sec,
            current_end_sec=observation.end_time_sec,
            current_evidence=observation.average_evidence,
            current_spatial=observation.majority_spatial_position,
            bed_context=self.state_engine.current_bed_context.value,
            state_history=list(self._state_history),
        )

        if prev_observation:
            ctx.previous_evidence = prev_observation.average_evidence
            ctx.previous_spatial = prev_observation.majority_spatial_position
            ctx.previous_activity = prev_activity.value if prev_activity else None

        if next_observation:
            ctx.next_evidence = next_observation.average_evidence
            ctx.next_spatial = next_observation.majority_spatial_position

        # Include pending event info
        ctx.pending_exit_since_sec = self.event_engine.potential_exit_start_time
        ctx.pending_return_since_sec = self.event_engine.potential_return_start_time

        return ctx

    def _process_segment_through_engines(
        self,
        observation: Observation,
        segment_frames: list,
        previous_activity: ActivityState,
        prev_observation: Observation | None,
        next_observation: Observation | None,
    ) -> tuple[ActivityState, dict | None]:
        """
        Run a single observation through all engines.
        Returns (final_activity, event_or_none).
        """
        # 1. Temporal State Engine
        current_activity, current_context = self.state_engine.process_observation(
            observation
        )

        # 2. Agent investigation (uncertainty-driven)
        if self._should_investigate(current_activity, observation) and segment_frames:
            temporal_ctx = self._build_temporal_context(
                observation,
                current_activity,
                prev_observation,
                previous_activity,
                next_observation,
            )
            decision = self.agent.investigate(
                observation, segment_frames[:3], temporal_context=temporal_ctx
            )
            if (
                decision.confidence
                > self.config.temporal_state.confidence_threshold
            ):
                current_activity = decision.confirmed_state
                self.state_engine.current_activity = current_activity
                logger.info(
                    f"Agent overrode state to {current_activity.value} "
                    f"(conf={decision.confidence:.2f}, vlm={decision.vlm_used})"
                )

        # 3. Record state history
        self._state_history.append(
            StateHistoryEntry(
                time_sec=observation.end_time_sec,
                activity=current_activity.value,
                bed_context=current_context.value,
            )
        )

        # 4. Bed Event Engine
        new_context, event = self.event_engine.process(
            current_activity, current_context, observation, previous_activity
        )
        if new_context != current_context:
            self.state_engine.current_bed_context = new_context
            current_context = new_context

        # 5. Timeline Engine
        self.timeline_engine.process_segment(
            observation.start_time_sec,
            observation.end_time_sec,
            current_activity,
            current_context,
            event,
        )

        # 6. Safety Engine
        safety = self.safety_engine.evaluate(
            current_activity, current_context, self.timeline_engine
        )

        # Update event safety decision if applicable
        if event:
            event.decision = safety.value

        return current_activity, event

    def run(self, video_path: str, output_path: str | None = None, visualize: bool = False) -> dict:
        """
        Run the full pipeline on a video file.
        Returns the FinalReport as a dictionary.
        """
        logger.info(f"Starting pipeline on: {video_path}")

        # 1. Video Ingestion
        ingester = VideoIngester(video_path)
        metadata = ingester.get_metadata()
        logger.info(
            f"Video: {metadata.width}x{metadata.height}, {metadata.fps}fps, {metadata.duration_sec:.1f}s"
        )

        target_fps = self.config.video.frame_sample_rate
        segment_duration = self.config.video.segment_duration_sec

        previous_activity = ActivityState.UNKNOWN
        prev_observation: Observation | None = None
        segment_count = 0

        # 2. Process each temporal segment with 1-segment look-ahead buffer
        for segment in ingester.generate_segments(target_fps, segment_duration):
            segment_count += 1

            # Build perception for this segment
            raw_frames = [f.image for f in segment.frames if f.image is not None]
            observation = self._build_observation(segment, raw_frames, visualize, previous_activity)

            # If there's a buffered segment waiting, process it now
            # (it now has the current segment as its "next" context)
            if self._buffered_segment is not None:
                buf = self._buffered_segment
                previous_activity, _ = self._process_segment_through_engines(
                    buf["observation"],
                    buf["frames"],
                    buf["previous_activity"],
                    buf["prev_observation"],
                    next_observation=observation,  # genuine look-ahead
                )
                prev_observation = buf["observation"]

            # Buffer the current segment for look-ahead processing
            self._buffered_segment = {
                "observation": observation,
                "frames": raw_frames,
                "previous_activity": previous_activity,
                "prev_observation": prev_observation,
            }

            if segment_count % 50 == 0:
                logger.info(
                    f"Processed {segment_count} segments ({observation.end_time_sec:.1f}s / {metadata.duration_sec:.1f}s)"
                )

        # 3. Flush the final buffered segment (no next context available)
        if self._buffered_segment is not None:
            buf = self._buffered_segment
            previous_activity, _ = self._process_segment_through_engines(
                buf["observation"],
                buf["frames"],
                buf["previous_activity"],
                buf["prev_observation"],
                next_observation=None,  # last segment, no future context
            )

        # 4. Finalize report
        if visualize:
            import cv2
            cv2.destroyAllWindows()
            
        logger.info(f"Pipeline complete. Processed {segment_count} segments.")
        report = self.timeline_engine.finalize()
        report.observation_duration_sec = metadata.duration_sec
        report.safety_decision = self.safety_engine.current_decision.value

        report_dict = report.model_dump()

        # 5. Write output
        if output_path:
            out = Path(output_path)
            out.parent.mkdir(parents=True, exist_ok=True)
            with open(out, "w") as f:
                json.dump(report_dict, f, indent=2, default=str)
            logger.info(f"Report written to {output_path}")

        return report_dict
