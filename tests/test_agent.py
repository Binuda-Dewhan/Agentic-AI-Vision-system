from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from sleep_monitor.agent.investigator import InvestigationAgent
from sleep_monitor.agent.schemas import AgentDecision
from sleep_monitor.config.settings import VlmConfig
from sleep_monitor.schemas.perception import ActivityEvidence, Observation
from sleep_monitor.schemas.state import ActivityState


@pytest.fixture
def vlm_config():
    return VlmConfig(
        vlm_enabled=True, vlm_provider="anthropic", vlm_model="claude-3-5-sonnet-latest"
    )


def create_mock_obs():
    return Observation(
        segment_id=1,
        start_time_sec=0.0,
        end_time_sec=2.0,
        frames=[],
        other_persons_present=False,
        average_evidence=ActivityEvidence(lying=0.4, sitting=0.4),  # Ambiguous
        majority_spatial_position="INSIDE",
    )


def test_agent_disabled():
    config = VlmConfig(vlm_enabled=False)
    agent = InvestigationAgent(config)

    decision = agent.investigate(create_mock_obs(), [])
    assert decision.confirmed_state == ActivityState.UNKNOWN
    assert "disabled" in decision.reasoning


@patch("sleep_monitor.agent.investigator.ChatAnthropic")
@patch("sleep_monitor.perception.vlm.ChatAnthropic")
def test_agent_graph_execution(mock_vlm_llm, mock_agent_llm, vlm_config):
    # Setup VLM mock
    mock_vlm_structured = MagicMock()
    mock_vlm_structured.invoke.return_value = MagicMock(
        activity_state=ActivityState.LYING_IN_BED, reasoning="VLM thinks lying down."
    )
    mock_vlm_llm.return_value.with_structured_output.return_value = mock_vlm_structured

    # Setup Agent mock
    mock_agent_structured = MagicMock()
    mock_agent_structured.invoke.return_value = AgentDecision(
        confirmed_state=ActivityState.LYING_IN_BED,
        confidence=0.85,
        reasoning="Synthesized VLM and context.",
    )
    mock_agent_llm.return_value.with_structured_output.return_value = (
        mock_agent_structured
    )

    agent = InvestigationAgent(vlm_config)

    frames = [np.zeros((10, 10, 3), dtype=np.uint8)]
    decision = agent.investigate(create_mock_obs(), frames)

    assert decision.confirmed_state == ActivityState.LYING_IN_BED
    assert decision.confidence == 0.85
    assert "Synthesized" in decision.reasoning

    # Verify graph nodes were hit by checking if LLM was invoked
    mock_agent_structured.invoke.assert_called_once()
