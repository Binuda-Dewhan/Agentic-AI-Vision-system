import math
from collections import Counter

from sleep_monitor.schemas.perception import (
    ActivityEvidence,
    FrameObservation,
    Observation,
    PoseKeypoints,
    SpatialFeatures,
)
from sleep_monitor.schemas.video import TemporalSegment


class ObservationBuilder:
    """Builds an aggregated Observation from a temporal segment."""

    def build_evidence(
        self,
        pose: PoseKeypoints | None,
        spatial: SpatialFeatures | None,
        movement_mag: float,
    ) -> ActivityEvidence:
        """Score activity evidence based on features."""
        evidence = ActivityEvidence(movement_magnitude=movement_mag)

        if not pose or not spatial:
            return evidence

        # Base logic for activity scoring
        is_moving = movement_mag > 20.0

        if spatial.position_relative_to_bed in ["INSIDE", "ON_EDGE"]:
            if pose.orientation == "HORIZONTAL":
                evidence.lying = 0.9 if not is_moving else 0.7
                evidence.sitting = 0.2
            elif pose.orientation == "UPRIGHT":
                evidence.sitting = 0.8
                evidence.standing = 0.4
            elif pose.orientation == "INTERMEDIATE":
                evidence.lying = 0.5
                evidence.sitting = 0.5
        elif spatial.position_relative_to_bed == "OUTSIDE":
            if pose.orientation == "UPRIGHT":
                if is_moving:
                    evidence.walking = 0.9
                    evidence.standing = 0.3
                else:
                    evidence.standing = 0.8
                    evidence.sitting = 0.4  # Could be sitting in a chair
            elif pose.orientation == "HORIZONTAL":
                # Fall risk?
                evidence.lying = 0.8
            else:
                evidence.sitting = 0.6
                evidence.standing = 0.4
        else:
            # UNCONFIGURED bed
            if pose.orientation == "HORIZONTAL":
                evidence.lying = 0.8
            elif pose.orientation == "UPRIGHT":
                if is_moving:
                    evidence.walking = 0.8
                else:
                    evidence.standing = 0.6
                    evidence.sitting = 0.6

        return evidence

    def build_observation(
        self,
        segment: TemporalSegment,
        frame_observations: list[FrameObservation],
        other_persons_present: bool,
    ) -> Observation:
        """Aggregate frame observations into a single segment observation."""

        # Calculate movement magnitude across frames
        valid_frames = [f for f in frame_observations if f.bbox is not None]
        for i in range(len(valid_frames)):
            f = valid_frames[i]
            if i > 0:
                prev = valid_frames[i - 1]
                dx = f.bbox.center_x - prev.bbox.center_x
                dy = f.bbox.center_y - prev.bbox.center_y
                movement = math.sqrt(dx**2 + dy**2)
            else:
                movement = 0.0

            f.evidence = self.build_evidence(f.pose, f.spatial, movement)

        # Aggregate evidence
        avg_evidence = ActivityEvidence()
        if valid_frames:
            avg_evidence.lying = sum(f.evidence.lying for f in valid_frames) / len(
                valid_frames
            )
            avg_evidence.sitting = sum(f.evidence.sitting for f in valid_frames) / len(
                valid_frames
            )
            avg_evidence.standing = sum(
                f.evidence.standing for f in valid_frames
            ) / len(valid_frames)
            avg_evidence.walking = sum(f.evidence.walking for f in valid_frames) / len(
                valid_frames
            )
            avg_evidence.movement_magnitude = sum(
                f.evidence.movement_magnitude for f in valid_frames
            ) / len(valid_frames)

        # Majority spatial
        positions = [
            f.spatial.position_relative_to_bed for f in valid_frames if f.spatial
        ]
        majority_spatial = (
            Counter(positions).most_common(1)[0][0] if positions else "UNKNOWN"
        )

        return Observation(
            segment_id=segment.segment_id,
            start_time_sec=segment.start_time_sec,
            end_time_sec=segment.end_time_sec,
            frames=frame_observations,
            other_persons_present=other_persons_present,
            average_evidence=avg_evidence,
            majority_spatial_position=majority_spatial,
        )
