import logging
from typing import TypedDict

import numpy as np
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, StateGraph

from sleep_monitor.agent.schemas import AgentDecision
from sleep_monitor.agent.temporal_context import TemporalContext
from sleep_monitor.config.settings import VlmConfig
from sleep_monitor.perception.vlm import VlmClassifier
from sleep_monitor.schemas.perception import Observation
from sleep_monitor.schemas.state import ActivityState

logger = logging.getLogger(__name__)


# Define the state for the LangGraph workflow
class AgentState(TypedDict):
    observation: Observation
    frames: list[np.ndarray]
    temporal_context: TemporalContext
    vlm_result: str
    historical_context: str
    vlm_was_used: bool
    decision: AgentDecision | None


class InvestigationAgent:
    """
    A LangGraph-based agent that investigates ambiguous temporal segments
    by combining rich temporal context with Vision-Language Model analysis.

    Decision path:
      START → GATHER TEMPORAL CONTEXT → ASSESS UNCERTAINTY
        → if context sufficient: deterministic/temporal decision
        → if context insufficient: VLM investigation
      → STRUCTURED DECISION → END

    Temporal context includes:
      - previous segment evidence & determined state
      - current segment evidence
      - next/following segment evidence (when available via buffer)
      - recent state history
      - current bed context & pending event candidates
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
        """
        Gather and format temporal context for the decision node.

        This node uses the TemporalContext object which contains:
        - previous segment evidence & determined activity
        - current segment evidence & spatial position
        - next/following segment evidence (when buffered look-ahead exists)
        - recent state history (last N transitions)
        - current bed context & pending event candidates
        """
        temporal_ctx = state["temporal_context"]
        formatted = temporal_ctx.format_for_prompt()
        return {"historical_context": formatted}

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
            "You have access to temporal context from surrounding video segments and recent state history.\n"
            "Use this temporal evidence to determine the most likely ActivityState.\n"
            "If the evidence is genuinely insufficient, return UNKNOWN — do not force a classification.\n\n"
            f"Temporal Context:\n{state.get('historical_context', 'None')}\n\n"
            f"VLM Analysis:\n{state.get('vlm_result', 'None')}\n\n"
            "Synthesize ALL available temporal evidence to determine the ActivityState."
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
        self,
        observation: Observation,
        frames: list[np.ndarray],
        temporal_context: TemporalContext | None = None,
    ) -> AgentDecision:
        """
        Entry point to trigger the investigation graph.

        Args:
            observation: Current segment observation.
            frames: Raw video frames for VLM analysis.
            temporal_context: Rich temporal context from the pipeline buffer.
        """
        if not self.enabled:
            return AgentDecision(
                confirmed_state=ActivityState.UNKNOWN,
                confidence=0.0,
                decision_type="UNABLE_TO_RESOLVE",
                reasoning="Agent disabled.",
                vlm_used=False,
            )

        if temporal_context is None:
            temporal_context = TemporalContext(
                current_segment_id=observation.segment_id,
                current_start_sec=observation.start_time_sec,
                current_end_sec=observation.end_time_sec,
                current_evidence=observation.average_evidence,
                current_spatial=observation.majority_spatial_position,
            )

        initial_state = {
            "observation": observation,
            "frames": frames,
            "temporal_context": temporal_context,
            "vlm_result": "Not run",
            "historical_context": "",
            "vlm_was_used": False,
            "decision": None,
        }

        result = self.graph.invoke(initial_state)
        return result["decision"]
