import logging

import cv2
import numpy as np
from ultralytics import YOLO

from sleep_monitor.schemas.perception import BBox, TrackedPerson

logger = logging.getLogger(__name__)


class PersonDetector:
    def __init__(
        self, model_path: str = "yolov8n.pt", confidence_threshold: float = 0.5
    ):
        """Initialize the YOLO detector and tracker."""
        logger.info(f"Loading YOLO model from {model_path}")
        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold

    def detect_and_track(
        self, frame: np.ndarray, persist: bool = True
    ) -> list[TrackedPerson]:
        """
        Run detection and tracking on a single frame.
        Returns a list of TrackedPerson objects.
        """
        # Run tracking. class=0 limits to 'person'.
        # persist=True keeps tracking state across frames.
        results = self.model.track(
            frame,
            persist=persist,
            classes=[0],
            conf=self.confidence_threshold,
            verbose=False,
            tracker="bytetrack.yaml",
        )

        persons = []
        if len(results) > 0 and len(results[0].boxes) > 0:
            boxes = results[0].boxes

            for i in range(len(boxes)):
                # Fall back to ID 0 if ByteTrack didn't assign one yet
                # (happens on first frame, scene cuts, or brief occlusion)
                if boxes.id is not None:
                    track_id = int(boxes.id[i].item())
                else:
                    track_id = i  # Use detection index as fallback ID

                conf = float(boxes.conf[i].item())
                xyxy = boxes.xyxy[i].cpu().numpy()

                bbox = BBox(x1=xyxy[0], y1=xyxy[1], x2=xyxy[2], y2=xyxy[3])

                persons.append(
                    TrackedPerson(track_id=track_id, bbox=bbox, confidence=conf)
                )

        return persons

    def draw_annotations(
        self, frame: np.ndarray, persons: list[TrackedPerson]
    ) -> np.ndarray:
        """Render bounding boxes and IDs on the frame for debugging."""
        annotated = frame.copy()

        for p in persons:
            color = (0, 255, 0) if p.is_primary_target else (0, 0, 255)
            thickness = 2 if p.is_primary_target else 1

            x1, y1 = int(p.bbox.x1), int(p.bbox.y1)
            x2, y2 = int(p.bbox.x2), int(p.bbox.y2)

            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)

            label = f"ID:{p.track_id} {p.confidence:.2f}"
            if p.is_primary_target:
                label = f"TARGET {label}"

            cv2.putText(
                annotated,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                thickness,
            )

        return annotated
