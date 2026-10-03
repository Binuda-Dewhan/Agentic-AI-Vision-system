from pathlib import Path

import yaml
from pydantic import BaseModel


class GroundTruthSegment(BaseModel):
    start_time: float
    end_time: float
    state: str


class GroundTruthEvent(BaseModel):
    type: str
    time: float


class VideoAnnotation(BaseModel):
    video_file: str
    annotator: str
    segments: list[GroundTruthSegment]
    events: list[GroundTruthEvent]


def load_annotations(yaml_path: str | Path) -> VideoAnnotation:
    """Load ground truth annotations from a YAML file."""
    path = Path(yaml_path)
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    return VideoAnnotation(**data)
