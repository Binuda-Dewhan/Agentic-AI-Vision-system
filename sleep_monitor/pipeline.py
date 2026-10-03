import json
import logging
from pathlib import Path

from sleep_monitor.agent.investigator import InvestigationAgent
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
from sleep_monitor.schemas.perception import FrameObservation
from sleep_monitor.schemas.state import ActivityState
from sleep_monitor.video.ingestion import VideoIngester

logger = logging.getLogger(__name__)


class Pipeline:
    """
    End-to-end orchestrator that wires all engines together.

    Video → Perception → Observation → StateEngine → BedEventEngine
         → TimelineEngine → SafetyEngine → FinalReport
    """

    def __init__(self, config: Settings):
        self.config = config

        # Perception components
        self.detector = PersonDetector(
            model_path=config.perception.detection_model,
            confidence_threshold=config.perception.detection_confidence_threshold,
        )
        self.target_selector = TargetSelector(mode=config.target_person.mode)
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

    def run(self, video_path: str, output_path: str | None = None) -> dict:
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
        segment_count = 0

        # 2. Process each temporal segment
        for segment in ingester.generate_segments(target_fps, segment_duration):
            segment_count += 1
            frame_observations = []
            any_other_persons = False
            segment_frames = []  # Raw frames for potential agent use

            for frame in segment.frames:
                # Decode frame
                raw_frame = frame.image
                if raw_frame is None:
                    continue

                segment_frames.append(raw_frame)

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

            # 3. Build Observation from frame-level data
            observation = self.observation_builder.build_observation(
                segment, frame_observations, any_other_persons
            )

            # 4. Temporal State Engine
            current_activity, current_context = self.state_engine.process_observation(
                observation
            )

            # 5. Agent investigation on low confidence
            # (Check if state is UNKNOWN for too long)
            if current_activity == ActivityState.UNKNOWN and segment_frames:
                decision = self.agent.investigate(observation, segment_frames[:3])
                if (
                    decision.confidence
                    > self.config.temporal_state.confidence_threshold
                ):
                    current_activity = decision.confirmed_state
                    self.state_engine.current_activity = current_activity
                    logger.info(
                        f"Agent overrode state to {current_activity.value} (conf={decision.confidence:.2f})"
                    )

            # 6. Bed Event Engine
            new_context, event = self.event_engine.process(
                current_activity, current_context, observation, previous_activity
            )
            if new_context != current_context:
                self.state_engine.current_bed_context = new_context
                current_context = new_context

            # 7. Timeline Engine
            self.timeline_engine.process_segment(
                observation.start_time_sec,
                observation.end_time_sec,
                current_activity,
                current_context,
                event,
            )

            # 8. Safety Engine
            safety = self.safety_engine.evaluate(
                current_activity, current_context, self.timeline_engine
            )

            # Update event safety decision if applicable
            if event:
                event.decision = safety.value

            previous_activity = current_activity

            if segment_count % 50 == 0:
                logger.info(
                    f"Processed {segment_count} segments ({observation.end_time_sec:.1f}s / {metadata.duration_sec:.1f}s)"
                )

        # 9. Finalize report
        logger.info(f"Pipeline complete. Processed {segment_count} segments.")
        report = self.timeline_engine.finalize()
        report.observation_duration_sec = metadata.duration_sec
        report.safety_decision = self.safety_engine.current_decision.value

        report_dict = report.model_dump()

        # 10. Write output
        if output_path:
            out = Path(output_path)
            out.parent.mkdir(parents=True, exist_ok=True)
            with open(out, "w") as f:
                json.dump(report_dict, f, indent=2, default=str)
            logger.info(f"Report written to {output_path}")

        return report_dict
