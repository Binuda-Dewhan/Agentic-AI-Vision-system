import logging
import math

from sleep_monitor.schemas.perception import FramePerception, TrackedPerson

logger = logging.getLogger(__name__)


class TargetSelector:
    def __init__(self, mode: str = "auto", reassociation_distance_px: int = 100):
        """
        Initialize target selector.
        Supported modes: 'auto' (first stable subject)
        """
        self.mode = mode
        self.primary_track_id: int | None = None
        self.last_known_bbox = None
        self.reassociation_distance_px = reassociation_distance_px

        # Used to require a person to be seen a few times before locking as target
        self._candidates: dict[int, int] = {}
        self._stable_threshold = 3  # frames

    def _distance(self, bbox1, bbox2) -> float:
        if bbox1 is None or bbox2 is None:
            return float("inf")
        dx = bbox1.center_x - bbox2.center_x
        dy = bbox1.center_y - bbox2.center_y
        return math.sqrt(dx**2 + dy**2)

    def select_target(
        self, frame_index: int, persons: list[TrackedPerson]
    ) -> FramePerception:
        """
        Identify the primary target among detected persons.
        Updates the is_primary_target flag and returns FramePerception.
        """
        # If no one is detected, just return
        if not persons:
            return FramePerception(
                frame_index=frame_index, persons=[], other_persons_present=False
            )

        if self.mode != "auto":
            # For future manual initialization
            return FramePerception(
                frame_index=frame_index,
                persons=persons,
                other_persons_present=len(persons) > 1,
            )

        # 1. Target not yet locked
        if self.primary_track_id is None:
            # Update candidates
            current_ids = {p.track_id for p in persons}
            for pid in current_ids:
                self._candidates[pid] = self._candidates.get(pid, 0) + 1

            # Check if any candidate is stable
            stable_candidates = [
                pid
                for pid, count in self._candidates.items()
                if count >= self._stable_threshold
            ]

            if stable_candidates:
                # Pick the one with the largest bounding box (closest to camera)
                # or just the first one if only one.
                best_person = max(
                    [p for p in persons if p.track_id in stable_candidates],
                    key=lambda p: p.bbox.area,
                    default=None,
                )

                if best_person:
                    self.primary_track_id = best_person.track_id
                    logger.info(
                        f"Target selected automatically: Track ID {self.primary_track_id}"
                    )

        # 2. Target locked, update persons
        target_found = False
        other_present = False

        for p in persons:
            if p.track_id == self.primary_track_id:
                p.is_primary_target = True
                self.last_known_bbox = p.bbox
                target_found = True
            else:
                other_present = True

        # 3. Disambiguation if primary track lost but others are present
        if (
            not target_found
            and self.primary_track_id is not None
            and self.last_known_bbox is not None
        ):
            # Has the primary ID changed due to tracking failure?
            # Check spatial proximity to last known location.
            best_match = None
            min_dist = float("inf")

            for p in persons:
                dist = self._distance(p.bbox, self.last_known_bbox)
                if (
                    dist < self.reassociation_distance_px and dist < min_dist
                ):  # Configurable threshold for spatial match
                    min_dist = dist
                    best_match = p

            if not best_match and len(persons) == 1:
                best_match = persons[0]
                logger.info(
                    f"Track ID changed {self.primary_track_id} -> {best_match.track_id} via single-person fallback."
                )
            elif best_match:
                logger.info(
                    f"Track ID changed {self.primary_track_id} -> {best_match.track_id} due to spatial proximity."
                )

            if best_match:
                self.primary_track_id = best_match.track_id
                best_match.is_primary_target = True
                self.last_known_bbox = best_match.bbox
                other_present = len(persons) > 1

        return FramePerception(
            frame_index=frame_index,
            persons=persons,
            other_persons_present=other_present,
        )
