from pathlib import Path

import yaml
from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

# Load .env variables into os.environ so Langchain/Google can see them
load_dotenv()


class TargetPersonConfig(BaseSettings):
    mode: str = "auto"
    reassociation_distance_px: int = 500  # Max pixel distance for re-associating a lost track. Increased to 500 for low FPS walking.


class VideoConfig(BaseSettings):
    frame_sample_rate: int = 2
    segment_duration_sec: int = 2


class PerceptionConfig(BaseSettings):
    detection_model: str = "yolov8n.pt"
    pose_model: str = "yolov8n-pose.pt"
    detection_confidence_threshold: float = (
        0.3  # Lowered from 0.5 — sleeping/covered persons score lower
    )
    keypoint_visibility_threshold: float = 0.5


class BedRegionConfig(BaseSettings):
    bed_region_polygon: list[tuple[int, int]] | None = None
    bed_edge_margin: int = 20


class TemporalStateConfig(BaseSettings):
    state_confirmation_window_sec: int = 4
    state_transition_hysteresis: float = 1.5
    confidence_threshold: float = 0.6


class BedEventsConfig(BaseSettings):
    bed_exit_hysteresis_sec: int = 10
    return_hysteresis_sec: int = 5


class SafetyRulesConfig(BaseSettings):
    monitor_bed_edge_duration_sec: int = 120
    out_of_bed_monitor_sec: int = 300
    alert_out_of_bed_duration_sec: int = 900
    alert_unknown_duration_sec: int = 300
    max_exits_before_alert: int = 3


class VlmConfig(BaseSettings):
    vlm_provider: str = "google"
    vlm_model: str = "gemini-3.8-flash"
    vlm_max_frames_per_query: int = 5
    vlm_enabled: bool = True


class Settings(BaseSettings):
    target_person: TargetPersonConfig = TargetPersonConfig()
    video: VideoConfig = VideoConfig()
    perception: PerceptionConfig = PerceptionConfig()
    bed_region: BedRegionConfig = BedRegionConfig()
    temporal_state: TemporalStateConfig = TemporalStateConfig()
    bed_events: BedEventsConfig = BedEventsConfig()
    safety_rules: SafetyRulesConfig = SafetyRulesConfig()
    vlm: VlmConfig = VlmConfig()

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        extra="ignore",
    )

    @classmethod
    def from_yaml(cls, yaml_path: str | Path) -> "Settings":
        path = Path(yaml_path)
        if not path.exists():
            return cls()
        with open(path, "r") as f:
            config_dict = yaml.safe_load(f) or {}
        return cls(**config_dict)


settings = Settings()
