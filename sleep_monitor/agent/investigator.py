import logging
from typing import TypedDict

import numpy as np
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, StateGraph

from sleep_monitor.agent.schemas import AgentDecision
from sleep_monitor.config.settings import VlmConfig
from sleep_monitor.perception.vlm import VlmClassifier
from sleep_monitor.schemas.perception import Observation
from sleep_monitor.schemas.state import ActivityState

logger = logging.getLogger(__name__)


# Define the state for the LangGraph workflow
class AgentState(TypedDict):
    observation: Observation
    frames: list[np.ndarray]
    vlm_result: str
    historical_context: str
    vlm_was_used: bool
    decision: AgentDecision | None


class InvestigationAgent:
    """
    A LangGraph-based agent that investigates ambiguous temporal segments
    by combining historical context with Vision-Language Model analysis.
    """

    def __init__(self, vlm_config: VlmConfig):
        self.vlm_classifier = VlmClassifier(vlm_config)
        self.enabled = vlm_config.vlm_enabled

        if self.enabled:
            try:
                self.llm = ChatGoogleGenerativeAI(
                    model=vlm_config.vlm_model,
                    temperature=0.0,
                    max_retries=0,  # Fail fast on quota errors; pipeline degrades to UNKNOWN
                )
                self.structured_llm = self.llm.with_structured_output(AgentDecision)
            except Exception as e:
                logger.error(f"Failed to initialize Agent LLM: {e}")
                self.enabled = False
                self.llm = None
                self.structured_llm = None
        else:
            self.llm = None
            self.structured_llm = None

        self.graph = self._build_graph()

    def _build_graph(self):
        builder = StateGraph(AgentState)

        # Add nodes
        builder.add_node("gather_context", self._node_gather_context)
        builder.add_node("run_vlm", self._node_run_vlm)
        builder.add_node("decide", self._node_decide)

        # Define edges and conditionals
        builder.set_entry_point("gather_context")

        builder.add_conditional_edges(
            "gather_context", self._should_run_vlm, {True: "run_vlm", False: "decide"}
        )

        builder.add_edge("run_vlm", "decide")
        builder.add_edge("decide", END)

        return builder.compile()

    def _node_gather_context(self, state: AgentState) -> dict:
        obs = state["observation"]
        hist = (
            f"Segment {obs.segment_id} ({obs.start_time_sec}s - {obs.end_time_sec}s).\n"
        )
        hist += f"Deterministic Evidence: Lying={obs.average_evidence.lying:.2f}, Sitting={obs.average_evidence.sitting:.2f}, Standing={obs.average_evidence.standing:.2f}.\n"
        hist += f"Spatial Position: {obs.majority_spatial_position}."

        return {"historical_context": hist}

    def _should_run_vlm(self, state: AgentState) -> bool:
        return self.enabled and len(state.get("frames", [])) > 0

    def _node_run_vlm(self, state: AgentState) -> dict:
        response = self.vlm_classifier.classify_frames(state["frames"])
        if response:
            res_str = f"VLM suggests {response.activity_state.value}. Reasoning: {response.reasoning}"
        else:
            res_str = "VLM analysis failed or was unavailable."
        return {"vlm_result": res_str, "vlm_was_used": True}

    def _node_decide(self, state: AgentState) -> dict:
        vlm_was_used = state.get("vlm_was_used", False)

        if not self.enabled or not self.structured_llm:
            return {
                "decision": AgentDecision(
                    confirmed_state=ActivityState.UNKNOWN,
                    confidence=0.0,
                    decision_type="UNABLE_TO_RESOLVE",
                    reasoning="Agent is disabled.",
                    vlm_used=vlm_was_used,
                )
            }

        prompt = (
            "You are the final decision node in an investigation agent for elderly monitoring.\n"
            f"Historical Context:\n{state.get('historical_context', 'None')}\n\n"
            f"VLM Analysis:\n{state.get('vlm_result', 'None')}\n\n"
            "Synthesize this information to determine the most likely ActivityState."
        )

        try:
            logger.info("Agent computing final decision...")
            decision = self.structured_llm.invoke(prompt)
            decision.vlm_used = vlm_was_used
            return {"decision": decision}
        except Exception as e:
            logger.error(f"Agent decision failed: {e}")
            return {
                "decision": AgentDecision(
                    confirmed_state=ActivityState.UNKNOWN,
                    confidence=0.0,
                    decision_type="UNABLE_TO_RESOLVE",
                    reasoning=f"LLM Error: {e!s}",
                    vlm_used=vlm_was_used,
                )
            }

    def investigate(
        self, observation: Observation, frames: list[np.ndarray]
    ) -> AgentDecision:
        """Entry point to trigger the graph."""
        if not self.enabled:
            return AgentDecision(
                confirmed_state=ActivityState.UNKNOWN,
                confidence=0.0,
                decision_type="UNABLE_TO_RESOLVE",
                reasoning="Agent disabled.",
                vlm_used=False,
            )

        initial_state = {
            "observation": observation,
            "frames": frames,
            "vlm_result": "Not run",
            "historical_context": "",
            "vlm_was_used": False,
            "decision": None,
        }

        result = self.graph.invoke(initial_state)
        return result["decision"]
