import logging
import math

import numpy as np
from ultralytics import YOLO

from sleep_monitor.schemas.perception import BBox, PoseKeypoints

logger = logging.getLogger(__name__)


class PoseEstimator:
    """Estimates pose keypoints and body orientation using YOLO-Pose."""

    def __init__(self, model_path: str = "yolov8n-pose.pt"):
        logger.info(f"Loading YOLO-Pose model from {model_path}")
        self.model = YOLO(model_path)

    def estimate(self, frame: np.ndarray, bbox: BBox) -> PoseKeypoints:
        """Extract keypoints for a person in the given bounding box."""
        h, w = frame.shape[:2]
        margin = 20
        x1 = max(0, int(bbox.x1) - margin)
        y1 = max(0, int(bbox.y1) - margin)
        x2 = min(w, int(bbox.x2) + margin)
        y2 = min(h, int(bbox.y2) + margin)

        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return PoseKeypoints(keypoints=[], orientation="UNKNOWN")

        results = self.model(crop, verbose=False)

        if (
            len(results) == 0
            or len(results[0].keypoints) == 0
            or results[0].keypoints.data.shape[1] == 0
        ):
            return PoseKeypoints(keypoints=[], orientation="UNKNOWN")

        # Get highest confidence person in crop
        kpts = results[0].keypoints.data[0].cpu().numpy()

        # Adjust keypoints back to original frame coordinates
        adjusted_kpts = []
        for kp in kpts:
            x, y, conf = kp
            if conf > 0.1:
                adjusted_kpts.append((float(x + x1), float(y + y1), float(conf)))
            else:
                adjusted_kpts.append((0.0, 0.0, 0.0))

        orientation = self._classify_orientation(adjusted_kpts)

        return PoseKeypoints(keypoints=adjusted_kpts, orientation=orientation)

    def _classify_orientation(
        self,
        keypoints: list[tuple[float, float, float]],
        bbox: "BBox | None" = None,
    ) -> str:
        """
        Classify body orientation using multi-signal approach.

        The core problem with shoulder→hip angle alone:
        A person lying in bed filmed from the foot/side of the bed appears
        as a *tall, narrow* shape in the camera frame — giving a large vertical
        (dy) component that fools a simple angle test into reporting UPRIGHT.

        We solve this with three corroborating signals:
          1. Full-body span: head→ankle vector angle (covers more body length)
          2. Shoulder-to-hip angle (original signal)
          3. Bbox aspect ratio — a lying person has either very tall bbox
             (side-camera) or very wide bbox (overhead camera). We use the
             full-body keypoint spread to detect the actual body axis.
        """
        if len(keypoints) < 13:
            return "UNKNOWN"

        ls = keypoints[5]  # left shoulder
        rs = keypoints[6]  # right shoulder
        lh = keypoints[11]  # left hip
        rh = keypoints[12]  # right hip

        # ── Signal 1: shoulder→hip angle ─────────────────────────────────
        shoulder_hip_angle = None
        if min(ls[2], rs[2], lh[2], rh[2]) >= 0.3:
            shoulder_x = (ls[0] + rs[0]) / 2
            shoulder_y = (ls[1] + rs[1]) / 2
            hip_x = (lh[0] + rh[0]) / 2
            hip_y = (lh[1] + rh[1]) / 2
            dy = abs(shoulder_y - hip_y)
            dx = abs(shoulder_x - hip_x)
            if dy > 0 or dx > 0:
                shoulder_hip_angle = math.degrees(math.atan2(dy, dx))

        # ── Signal 2: full-body span (nose→ankle) ─────────────────────────
        # Use all high-confidence keypoints to find the primary body axis
        full_body_angle = None
        nose = keypoints[0] if len(keypoints) > 0 else None
        lank = keypoints[15] if len(keypoints) > 15 else None
        rank = keypoints[16] if len(keypoints) > 16 else None

        top_kp = nose if nose and nose[2] >= 0.3 else None
        # Prefer the ankle with higher confidence
        if lank and rank:
            bot_kp = lank if lank[2] >= rank[2] else rank
            if bot_kp[2] < 0.3:
                bot_kp = None
        elif lank and lank[2] >= 0.3:
            bot_kp = lank
        elif rank and rank[2] >= 0.3:
            bot_kp = rank
        else:
            bot_kp = None

        if top_kp and bot_kp:
            span_dy = abs(bot_kp[1] - top_kp[1])
            span_dx = abs(bot_kp[0] - top_kp[0])
            if span_dy > 0 or span_dx > 0:
                full_body_angle = math.degrees(math.atan2(span_dy, span_dx))

        # ── Signal 3: keypoint spread ratio ──────────────────────────────
        # Compare horizontal vs vertical spread across ALL confident keypoints
        # If horizontal spread dominates → body is lying across the frame (overhead view)
        # If neither dominates clearly → use other signals
        conf_kpts = [(kp[0], kp[1]) for kp in keypoints if kp[2] >= 0.3]
        keypoint_axis = None
        if len(conf_kpts) >= 4:
            xs = [p[0] for p in conf_kpts]
            ys = [p[1] for p in conf_kpts]
            h_spread = max(xs) - min(xs)
            v_spread = max(ys) - min(ys)
            if h_spread > 0 and v_spread > 0:
                ratio = h_spread / v_spread
                # ratio >> 1 → wide horizontal body (overhead lying)
                # ratio << 1 → tall vertical body (side-camera lying OR standing)
                # We only use this signal when it's clearly horizontal
                if ratio > 1.8:
                    keypoint_axis = "HORIZONTAL"
                # ratio < 0.6 is ambiguous (could be either lying-from-side or upright)

        # ── Decision: vote across signals ─────────────────────────────────
        # Angles > 50° → UPRIGHT (body mostly vertical relative to its own axis)
        # Angles < 35° → HORIZONTAL (body mostly horizontal)
        # 35–50°       → INTERMEDIATE
        #
        # CRITICAL INSIGHT: The full-body angle IS the most reliable single
        # signal. If both full_body_angle AND shoulder_hip_angle agree, trust them.
        # If keypoint_axis says HORIZONTAL, that overrides an ambiguous angle.

        primary_angle = (
            full_body_angle if full_body_angle is not None else shoulder_hip_angle
        )

        if primary_angle is None:
            return "UNKNOWN"

        if keypoint_axis == "HORIZONTAL":
            return "HORIZONTAL"

        # For the side-camera lying case: the body appears vertical in frame
        # (large dy in full_body_angle) but the person IS lying.
        # The key discriminator: a STANDING person has hips directly below
        # shoulders. A LYING person filmed from the side has hips offset LATERALLY
        # from shoulders (large dx in shoulder→hip vector relative to body height).
        if shoulder_hip_angle is not None and full_body_angle is not None:
            # If the full-body span is largely vertical but shoulder→hip has
            # significant horizontal offset, this indicates lying on side
            if full_body_angle > 50 and shoulder_hip_angle < 60:
                # Check if hips are laterally displaced from shoulders —
                # lateral displacement > 20% of body height indicates lying
                if min(ls[2], rs[2], lh[2], rh[2]) >= 0.3:
                    s_x = (ls[0] + rs[0]) / 2
                    h_x = (lh[0] + rh[0]) / 2
                    s_y = (ls[1] + rs[1]) / 2
                    h_y = (lh[1] + rh[1]) / 2
                    body_height_px = abs(h_y - s_y) + 1e-6
                    lateral_ratio = abs(h_x - s_x) / body_height_px
                    if lateral_ratio > 0.25:
                        return "HORIZONTAL"

        if primary_angle > 50:
            return "UPRIGHT"
        elif primary_angle < 35:
            return "HORIZONTAL"
        else:
            return "INTERMEDIATE"
