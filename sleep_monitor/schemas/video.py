import numpy as np
from pydantic import BaseModel, ConfigDict


class VideoMetadata(BaseModel):
    """Metadata extracted from a video file."""

    width: int
    height: int
    fps: float
    total_frames: int
    duration_sec: float
    codec: str | None = None


class Frame(BaseModel):
    """A single sampled video frame with its temporal metadata."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    frame_index: int
    timestamp_sec: float
    image: np.ndarray  # BGR numpy array from OpenCV


class TemporalSegment(BaseModel):
    """A temporal window containing multiple sampled frames."""

    segment_id: int
    start_time_sec: float
    end_time_sec: float
    frames: list[Frame]
