import cv2
import numpy as np
import pytest

from sleep_monitor.video.ingestion import VideoIngester


@pytest.fixture
def dummy_video(tmp_path):
    """Create a short dummy video file for testing."""
    video_path = tmp_path / "test_video.mp4"

    # 30 fps, 2 seconds long -> 60 frames
    fps = 30
    width, height = 640, 480
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(video_path), fourcc, fps, (width, height))

    for i in range(60):
        # Create a frame with a moving square
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        x = i * 10
        cv2.rectangle(frame, (x, 100), (x + 50, 150), (0, 255, 0), -1)
        out.write(frame)

    out.release()
    return video_path


def test_video_metadata(dummy_video):
    ingester = VideoIngester(dummy_video)
    metadata = ingester.get_metadata()

    assert metadata.width == 640
    assert metadata.height == 480
    assert metadata.fps == 30.0
    assert metadata.total_frames == 60
    assert metadata.duration_sec == 2.0


def test_video_segments(dummy_video):
    ingester = VideoIngester(dummy_video)

    # Target 2 fps, 1 second segments -> 2 segments, 2 frames each
    segments = list(
        ingester.generate_segments(target_fps=2.0, segment_duration_sec=1.0)
    )

    assert len(segments) == 2

    # First segment
    assert segments[0].segment_id == 0
    assert len(segments[0].frames) == 2

    # Second segment
    assert segments[1].segment_id == 1
    assert len(segments[1].frames) == 2

    # Verify timestamps (frame 0 -> 0.0s, frame 15 -> 0.5s, frame 30 -> 1.0s, frame 45 -> 1.5s)
    # With fps 30 and target 2, interval is 15 frames.
    # Seg 0 frames: idx 0 (0.0s), idx 15 (0.5s)
    assert segments[0].frames[0].frame_index == 0
    assert segments[0].frames[0].timestamp_sec == 0.0
    assert segments[0].frames[1].frame_index == 15
    assert segments[0].frames[1].timestamp_sec == 0.5

    # Seg 1 frames: idx 30 (1.0s), idx 45 (1.5s)
    assert segments[1].frames[0].frame_index == 30
    assert segments[1].frames[0].timestamp_sec == 1.0
    assert segments[1].frames[1].frame_index == 45
    assert segments[1].frames[1].timestamp_sec == 1.5
