import logging
from collections.abc import Generator
from pathlib import Path

import cv2

from sleep_monitor.schemas.video import Frame, TemporalSegment, VideoMetadata

logger = logging.getLogger(__name__)


class VideoIngester:
    def __init__(self, video_path: str | Path):
        self.video_path = str(Path(video_path).resolve())
        if not Path(self.video_path).exists():
            raise FileNotFoundError(f"Video file not found: {self.video_path}")

    def get_metadata(self) -> VideoMetadata:
        """Extract metadata from the video."""
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise ValueError(f"Failed to open video: {self.video_path}")

        try:
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = float(cap.get(cv2.CAP_PROP_FPS))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            fourcc_int = int(cap.get(cv2.CAP_PROP_FOURCC))
            codec = "".join([chr((fourcc_int >> 8 * i) & 0xFF) for i in range(4)])

            duration_sec = total_frames / fps if fps > 0 else 0.0

            return VideoMetadata(
                width=width,
                height=height,
                fps=fps,
                total_frames=total_frames,
                duration_sec=duration_sec,
                codec=codec,
            )
        finally:
            cap.release()

    def generate_segments(
        self, target_fps: float = 2.0, segment_duration_sec: float = 2.0
    ) -> Generator[TemporalSegment, None, None]:
        """
        Yield temporal segments from the video.
        Samples frames at target_fps and groups them into segments of segment_duration_sec.

        Segment end_time represents the temporal COVERAGE of the segment
        (start_time + segment_duration), not the timestamp of the last sampled
        frame. This guarantees:
          - adjacent segments are contiguous (no gaps)
          - segment durations sum to the video duration
          - the final partial segment is clamped to video duration
        """
        metadata = self.get_metadata()
        cap = cv2.VideoCapture(self.video_path)

        if not cap.isOpened():
            raise ValueError(f"Failed to open video: {self.video_path}")

        # We need to sample `target_fps` frames every second.
        # This means we want a frame every `1.0 / target_fps` seconds.
        # In terms of original frames, that's every `original_fps / target_fps` frames.
        frame_interval = max(1, int(round(metadata.fps / target_fps)))

        frames_per_segment = max(1, int(round(target_fps * segment_duration_sec)))

        segment_id = 0
        current_segment_frames: list[Frame] = []

        frame_idx = 0
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_idx % frame_interval == 0:
                    timestamp_sec = frame_idx / metadata.fps

                    sampled_frame = Frame(
                        frame_index=frame_idx, timestamp_sec=timestamp_sec, image=frame
                    )
                    current_segment_frames.append(sampled_frame)

                    if len(current_segment_frames) == frames_per_segment:
                        start_time = current_segment_frames[0].timestamp_sec
                        # End time = start of the NEXT segment (contiguous coverage)
                        end_time = min(
                            start_time + segment_duration_sec,
                            metadata.duration_sec,
                        )

                        yield TemporalSegment(
                            segment_id=segment_id,
                            start_time_sec=start_time,
                            end_time_sec=end_time,
                            frames=current_segment_frames,
                        )

                        segment_id += 1
                        current_segment_frames = []

                frame_idx += 1

            # Yield any remaining frames as the final segment
            if current_segment_frames:
                start_time = current_segment_frames[0].timestamp_sec
                # Final segment extends to video end
                end_time = metadata.duration_sec
                yield TemporalSegment(
                    segment_id=segment_id,
                    start_time_sec=start_time,
                    end_time_sec=end_time,
                    frames=current_segment_frames,
                )

        finally:
            cap.release()
