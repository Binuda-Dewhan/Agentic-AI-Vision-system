from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from sleep_monitor.config.settings import VlmConfig
from sleep_monitor.perception.vlm import VlmClassifier, VlmResponse
from sleep_monitor.schemas.state import ActivityState


@pytest.fixture
def vlm_config():
    return VlmConfig(
        vlm_provider="anthropic",
        vlm_model="claude-3-5-sonnet-latest",
        vlm_max_frames_per_query=2,
        vlm_enabled=True,
    )


def test_vlm_disabled():
    config = VlmConfig(vlm_enabled=False)
    classifier = VlmClassifier(config)

    assert classifier.enabled is False

    dummy_frame = np.zeros((100, 100, 3), dtype=np.uint8)
    response = classifier.classify_frames([dummy_frame])

    assert response.activity_state == ActivityState.UNKNOWN
    assert "disabled" in response.reasoning.lower()


@patch("sleep_monitor.perception.vlm.ChatAnthropic")
def test_vlm_enabled_mock_response(mock_chat_class, vlm_config):
    # Setup mock
    mock_llm = MagicMock()
    mock_structured = MagicMock()

    # Configure structured LLM to return a predefined response
    mock_structured.invoke.return_value = VlmResponse(
        activity_state=ActivityState.SITTING_ON_BED,
        reasoning="Person is upright on the bed.",
    )

    mock_llm.with_structured_output.return_value = mock_structured
    mock_chat_class.return_value = mock_llm

    # Initialize classifier
    classifier = VlmClassifier(vlm_config)
    assert classifier.enabled is True

    # Test with dummy frames
    dummy_frames = [
        np.zeros((100, 100, 3), dtype=np.uint8),
        np.zeros((100, 100, 3), dtype=np.uint8),
        np.zeros((100, 100, 3), dtype=np.uint8),  # Should truncate to max_frames=2
    ]

    response = classifier.classify_frames(dummy_frames)

    # Verify response
    assert response.activity_state == ActivityState.SITTING_ON_BED

    # Verify mock was called correctly
    mock_structured.invoke.assert_called_once()
    call_args = mock_structured.invoke.call_args[0][0]  # The messages list

    # System message + Human message
    assert len(call_args) == 2
    human_msg = call_args[1]

    # Text prompt + 2 image URLs
    assert len(human_msg.content) == 3
