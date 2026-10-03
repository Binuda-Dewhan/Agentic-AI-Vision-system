import base64
import logging

import cv2
import numpy as np
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

from sleep_monitor.config.settings import VlmConfig
from sleep_monitor.schemas.state import ActivityState

logger = logging.getLogger(__name__)


class VlmResponse(BaseModel):
    """Structured response from the Vision-Language Model."""

    activity_state: ActivityState = Field(
        description="The activity state of the person in the image. Choose UNKNOWN if unclear."
    )
    reasoning: str = Field(description="Brief reasoning for the chosen activity state.")


class VlmClassifier:
    """
    Acts as a semantic fallback classifier when deterministic models
    yield UNKNOWN or low-confidence results.
    """

    def __init__(self, config: VlmConfig):
        self.config = config
        self.enabled = config.vlm_enabled

        if self.enabled:
            # Note: Expects GOOGLE_API_KEY environment variable
            try:
                self.llm = ChatGoogleGenerativeAI(
                    model=self.config.vlm_model,
                    temperature=0.0,
                    max_retries=0,  # Fail fast on quota errors; pipeline degrades to UNKNOWN
                )
                self.structured_llm = self.llm.with_structured_output(VlmResponse)
            except Exception as e:
                logger.error(f"Failed to initialize VLM: {e}")
                self.enabled = False
                self.llm = None
                self.structured_llm = None
        else:
            self.llm = None
            self.structured_llm = None

    def _encode_image(self, frame: np.ndarray) -> str:
        """Resize image to save tokens, encode as base64."""
        h, w = frame.shape[:2]
        max_dim = 512
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            frame = cv2.resize(frame, (int(w * scale), int(h * scale)))

        _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        return base64.b64encode(buffer).decode("utf-8")

    def classify_frames(self, frames: list[np.ndarray]) -> VlmResponse | None:
        """
        Send frames to VLM to determine ActivityState.
        Returns None if disabled or if there are no frames.
        """
        if not self.enabled or not self.structured_llm:
            logger.info("VLM is disabled. Returning UNKNOWN.")
            return VlmResponse(
                activity_state=ActivityState.UNKNOWN, reasoning="VLM disabled."
            )

        if not frames:
            return None

        # Select up to max_frames
        selected_frames = frames[: self.config.vlm_max_frames_per_query]

        content = [
            {
                "type": "text",
                "text": "Analyze the sequence of frames and determine the activity state of the elderly person.",
            }
        ]

        for frame in selected_frames:
            b64_image = self._encode_image(frame)
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{b64_image}"},
                }
            )

        messages = [
            SystemMessage(
                content="You are an AI assistant analyzing security footage of an elderly person. "
                "Classify their activity state strictly from the allowed enum values: "
                "LYING_IN_BED, SITTING_ON_BED, SITTING_OUTSIDE_BED, STANDING, WALKING, UNKNOWN. "
                "Provide a brief reasoning."
            ),
            HumanMessage(content=content),
        ]

        try:
            logger.info(f"Querying VLM with {len(selected_frames)} frame(s)...")
            response = self.structured_llm.invoke(messages)
            return response
        except Exception as e:
            logger.error(f"VLM classification failed: {e}")
            return VlmResponse(
                activity_state=ActivityState.UNKNOWN, reasoning=f"Error: {e!s}"
            )
