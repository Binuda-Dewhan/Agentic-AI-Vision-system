from pydantic import BaseModel


class BBox(BaseModel):
    """Bounding box coordinates (xyxy)."""

    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def center_x(self) -> float:
        return (self.x1 + self.x2) / 2

    @property
    def center_y(self) -> float:
        return (self.y1 + self.y2) / 2

    @property
    def area(self) -> float:
        return (self.x2 - self.x1) * (self.y2 - self.y1)


class TrackedPerson(BaseModel):
    """A person detected and tracked in a frame."""

    track_id: int
    bbox: BBox
    confidence: float
    is_primary_target: bool = False


class FramePerception(BaseModel):
    """All perception data for a single frame."""

    frame_index: int
    persons: list[TrackedPerson]
    other_persons_present: bool = False


class PoseKeypoints(BaseModel):
    """Extracted pose keypoints (x, y, confidence)."""

    keypoints: list[tuple[float, float, float]]
    orientation: str  # "HORIZONTAL", "UPRIGHT", "INTERMEDIATE", "UNKNOWN"


class SpatialFeatures(BaseModel):
    """Spatial relationship to the bed region."""

    bed_overlap_ratio: float = 0.0
    distance_to_bed_center: float = -1.0
    position_relative_to_bed: str = (
        "UNKNOWN"  # "INSIDE", "ON_EDGE", "OUTSIDE", "UNCONFIGURED"
    )


class ActivityEvidence(BaseModel):
    """Evidence scores for different activity states [0.0, 1.0]."""

    lying: float = 0.0
    sitting: float = 0.0
    standing: float = 0.0
    walking: float = 0.0
    movement_magnitude: float = 0.0


class FrameObservation(BaseModel):
    """Perception data for the target person in a single frame."""

    frame_index: int
    timestamp_sec: float
    bbox: BBox | None = None
    pose: PoseKeypoints | None = None
    spatial: SpatialFeatures | None = None
    evidence: ActivityEvidence | None = None


class Observation(BaseModel):
    """Aggregated observation for a temporal segment."""

    segment_id: int
    start_time_sec: float
    end_time_sec: float
    frames: list[FrameObservation]
    other_persons_present: bool = False
    average_evidence: ActivityEvidence
    majority_spatial_position: str
